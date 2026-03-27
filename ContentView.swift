import SwiftUI
import ARKit
import SceneKit
import Vision
import Combine

// MARK: - Data Model

struct PersonDepthInfo: Identifiable {
    let id: Int
    let averageDepth: Float
    let minDepth: Float
    let maxDepth: Float
    let pixelCount: Int
    let normRect: CGRect
}

// MARK: - ViewModel

final class ARViewModel: ObservableObject {
    @Published var people: [PersonDepthInfo] = []
    @Published var statusMessage = "Initialising…"
    @Published var isRunning = false
}

// MARK: - UIViewRepresentable  (owns the ARSCNView + ARSession + delegate)

struct ARCameraView: UIViewRepresentable {

    @ObservedObject var vm: ARViewModel

    func makeCoordinator() -> Coordinator {
        Coordinator(vm: vm)
    }

    func makeUIView(context: Context) -> ARSCNView {
        let scnView = ARSCNView()
        scnView.scene = SCNScene()
        scnView.automaticallyUpdatesLighting = true

        scnView.session.delegate = context.coordinator
        context.coordinator.scnView = scnView

        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentationWithDepth) else {
            DispatchQueue.main.async {
                self.vm.statusMessage = "⚠️ Device does not support person segmentation with depth"
            }
            return scnView
        }

        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = [.personSegmentationWithDepth]
        scnView.session.run(config, options: [.resetTracking, .removeExistingAnchors])

        DispatchQueue.main.async {
            self.vm.isRunning = true
            self.vm.statusMessage = "Running * point at people"
        }

        return scnView
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}

    // MARK: - Coordinator

    final class Coordinator: NSObject, ARSessionDelegate {

        weak var scnView: ARSCNView?
        let vm: ARViewModel

        private var frameCounter = 0
        private let visionQueue = DispatchQueue(label: "vision.queue", qos: .userInitiated)
        private let humanRequest: VNDetectHumanRectanglesRequest
        private var isProcessingFrame = false
        private var nextTrackedID = 0

        private struct TrackedPerson {
            let id: Int
            let rect: CGRect
        }

        private var previousTrackedPeople: [TrackedPerson] = []

        init(vm: ARViewModel) {
            self.vm = vm
            let request = VNDetectHumanRectanglesRequest()
            request.upperBodyOnly = false
            self.humanRequest = request
            super.init()
        }

        // -----------------------------------------------------------------
        // MARK: ARSessionDelegate
        // -----------------------------------------------------------------
        func session(_ session: ARSession, didUpdate frame: ARFrame) {
            frameCounter = (frameCounter + 1) % 3
            guard frameCounter == 0 else { return }
            guard !isProcessingFrame else { return }

            guard frame.segmentationBuffer != nil,
                  frame.estimatedDepthData != nil else {
                DispatchQueue.main.async {
                    self.vm.statusMessage = "Waiting for seg+depth…"
                }
                return
            }

            isProcessingFrame = true

            visionQueue.async { [weak self] in
                guard let self else { return }
                defer { self.isProcessingFrame = false }

                let handler = VNImageRequestHandler(
                    cvPixelBuffer: frame.capturedImage,
                    orientation: .right,
                    options: [:]
                )

                do {
                    try handler.perform([self.humanRequest])
                    let observations = (self.humanRequest.results as? [VNHumanObservation]) ?? []
                    let result = self.extractPeople(from: frame, observations: observations)
                    let msg = result.isEmpty
                        ? "No people detected"
                        : "\(result.count) person\(result.count == 1 ? "" : "s") detected"

                    DispatchQueue.main.async {
                        self.vm.people = result
                        self.vm.statusMessage = msg
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.vm.statusMessage = "Vision error: \(error.localizedDescription)"
                    }
                }
            }
        }

        func session(_ session: ARSession, didFailWithError error: Error) {
            DispatchQueue.main.async { self.vm.statusMessage = "Error: \(error.localizedDescription)" }
        }

        func sessionWasInterrupted(_ session: ARSession) {
            DispatchQueue.main.async { self.vm.statusMessage = "Interrupted" }
        }

        func sessionInterruptionEnded(_ session: ARSession) {
            DispatchQueue.main.async { self.vm.statusMessage = "Resuming…" }
        }

        // -----------------------------------------------------------------
        // MARK: Core extraction – Vision rectangles + segmentationBuffer + estimatedDepthData
        // -----------------------------------------------------------------
        private func extractPeople(from frame: ARFrame, observations: [VNHumanObservation]) -> [PersonDepthInfo] {
            guard let segBuffer = frame.segmentationBuffer,
                  let depthBuffer = frame.estimatedDepthData else {
                return []
            }

            CVPixelBufferLockBaseAddress(segBuffer, .readOnly)
            CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
            defer {
                CVPixelBufferUnlockBaseAddress(segBuffer, .readOnly)
                CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)
            }

            guard let segBase = CVPixelBufferGetBaseAddress(segBuffer),
                  let depthBase = CVPixelBufferGetBaseAddress(depthBuffer) else {
                return []
            }

            let segW = CVPixelBufferGetWidth(segBuffer)
            let segH = CVPixelBufferGetHeight(segBuffer)
            let segBPR = CVPixelBufferGetBytesPerRow(segBuffer)

            let depW = CVPixelBufferGetWidth(depthBuffer)
            let depH = CVPixelBufferGetHeight(depthBuffer)
            let depBPR = CVPixelBufferGetBytesPerRow(depthBuffer)
            let depthStride = depBPR / MemoryLayout<Float32>.size

            let xScale = Float(depW) / Float(segW)
            let yScale = Float(depH) / Float(segH)

            let segPtr = segBase.assumingMemoryBound(to: UInt8.self)
            let depthPtr = depthBase.assumingMemoryBound(to: Float32.self)

            var rawPeople: [(rect: CGRect, avg: Float, min: Float, max: Float, count: Int)] = []

            for obs in observations {
                let bbox = obs.boundingBox

                let minX = max(0, Int(bbox.minX * CGFloat(segW)))
                let maxX = min(segW - 1, Int(bbox.maxX * CGFloat(segW)))

                let minYVision = max(0, Int(bbox.minY * CGFloat(segH)))
                let maxYVision = min(segH - 1, Int(bbox.maxY * CGFloat(segH)))

                // Vision bbox uses bottom-left origin; CVPixelBuffer uses top-left origin.
                let minY = segH - 1 - maxYVision
                let maxY = segH - 1 - minYVision

                guard minX <= maxX, minY <= maxY else { continue }

                var depthValues: [Float] = []
                depthValues.reserveCapacity(max(100, (maxX - minX + 1) * (maxY - minY + 1) / 5))

                for row in minY...maxY {
                    let rowBase = row * segBPR

                    for col in minX...maxX {
                        let segValue = segPtr[rowBase + col]
                        guard segValue != 0 else { continue }

                        let dc = min(depW - 1, Int(Float(col) * xScale))
                        let dr = min(depH - 1, Int(Float(row) * yScale))
                        let depth = depthPtr[dr * depthStride + dc]

                        guard depth.isFinite, depth > 0 else { continue }
                        depthValues.append(depth)
                    }
                }

                guard depthValues.count > 50 else { continue }

                depthValues.sort()
                let count = depthValues.count
                let median = depthValues[count / 2]
                let minDepth = depthValues.first ?? median
                let maxDepth = depthValues.last ?? median

                rawPeople.append((
                    rect: bbox,
                    avg: median,
                    min: minDepth,
                    max: maxDepth,
                    count: count
                ))
            }

            return assignStableIDs(to: rawPeople)
        }

        private func assignStableIDs(
            to rawPeople: [(rect: CGRect, avg: Float, min: Float, max: Float, count: Int)]
        ) -> [PersonDepthInfo] {
            var results: [PersonDepthInfo] = []
            var newTracked: [TrackedPerson] = []
            var usedPreviousIDs = Set<Int>()

            for person in rawPeople {
                let center = CGPoint(x: person.rect.midX, y: person.rect.midY)

                var bestID: Int?
                var bestDistance = CGFloat.greatestFiniteMagnitude

                for prev in previousTrackedPeople where !usedPreviousIDs.contains(prev.id) {
                    let prevCenter = CGPoint(x: prev.rect.midX, y: prev.rect.midY)
                    let dx = center.x - prevCenter.x
                    let dy = center.y - prevCenter.y
                    let dist = sqrt(dx * dx + dy * dy)

                    if dist < bestDistance, dist < 0.12 {
                        bestDistance = dist
                        bestID = prev.id
                    }
                }

                let assignedID: Int
                if let id = bestID {
                    assignedID = id
                    usedPreviousIDs.insert(id)
                } else {
                    assignedID = nextTrackedID
                    nextTrackedID += 1
                }

                newTracked.append(TrackedPerson(id: assignedID, rect: person.rect))

                results.append(
                    PersonDepthInfo(
                        id: assignedID,
                        averageDepth: person.avg,
                        minDepth: person.min,
                        maxDepth: person.max,
                        pixelCount: person.count,
                        normRect: person.rect
                    )
                )
            }

            previousTrackedPeople = newTracked
            return results.sorted { $0.id < $1.id }
        }
    }
}

// MARK: - Depth colour  (green = close → red = far, 0–8 m)

private func depthColor(_ metres: Float) -> Color {
    let t = Double(min(metres / 8.0, 1.0))
    return Color(hue: (1.0 - t) * 0.35, saturation: 1, brightness: 1)
}

// MARK: - Dot overlay

struct PersonDotOverlay: View {
    let people: [PersonDepthInfo]

    var body: some View {
        GeometryReader { geo in
            ForEach(people) { person in
                let r = person.normRect
                let cx = r.midX * geo.size.width
                let cy = (r.minY * geo.size.height) - 40
                let c = depthColor(person.averageDepth)

                ZStack {
                    Circle().fill(c.opacity(0.25)).frame(width: 48, height: 48)
                    Circle().fill(c).frame(width: 22, height: 22)
                        .shadow(color: c.opacity(0.9), radius: 8)
                    Text(String(format: "%.1fm", person.averageDepth))
                        .font(.system(size: 11, weight: .black, design: .monospaced))
                        .foregroundColor(.white)
                        .shadow(color: .black.opacity(0.8), radius: 2)
                        .offset(y: 30)
                }
                .position(x: cx, y: max(cy, 40))
            }
        }
    }
}

// MARK: - ContentView

struct ContentView: View {
    @StateObject private var vm = ARViewModel()

    var body: some View {
        ZStack(alignment: .bottom) {
            ARCameraView(vm: vm)
                .ignoresSafeArea()

            PersonDotOverlay(people: vm.people)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                HStack(spacing: 8) {
                    Image(systemName: vm.people.isEmpty ? "person.slash.fill" : "person.2.fill")
                        .foregroundColor(vm.people.isEmpty ? .gray : .green)
                    Text(vm.statusMessage)
                        .font(.subheadline.bold())
                        .foregroundColor(.white)
                    Spacer()
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(.ultraThinMaterial)

                if !vm.people.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 10) {
                            ForEach(vm.people) { person in
                                PersonChip(person: person)
                            }
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                    }
                    .background(.ultraThinMaterial)
                }
            }
        }
    }
}

// MARK: - PersonChip

private struct PersonChip: View {
    let person: PersonDepthInfo
    private var c: Color { depthColor(person.averageDepth) }

    var body: some View {
        HStack(spacing: 8) {
            Circle().fill(c).frame(width: 10, height: 10)
                .shadow(color: c.opacity(0.8), radius: 4)
            VStack(alignment: .leading, spacing: 2) {
                Text("Person \(person.id)")
                    .font(.caption2.bold()).foregroundColor(.white)
                Text(String(format: "%.2f m", person.averageDepth))
                    .font(.system(size: 15, weight: .bold, design: .monospaced))
                    .foregroundColor(c)
                Text(String(format: "%.1f – %.1f m", person.minDepth, person.maxDepth))
                    .font(.caption2).foregroundColor(.white.opacity(0.5))
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(Color.black.opacity(0.45))
        .cornerRadius(12)
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(c.opacity(0.6), lineWidth: 1))
    }
}
