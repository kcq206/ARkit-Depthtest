import SwiftUI
import ARKit
import SceneKit
import Vision
import UIKit
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

// MARK: - Overlay

final class PersonBoundingBoxOverlayView: UIView {
    private var dotViews: [Int: UIView] = [:]
    private var textLabels: [Int: UILabel] = [:]

    override init(frame: CGRect) {
        super.init(frame: frame)
        isUserInteractionEnabled = false
        backgroundColor = .clear
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func updatePeople(_ people: [PersonDepthInfo]) {
        let activeIDs = Set(people.map { $0.id })

        for (id, view) in dotViews where !activeIDs.contains(id) {
            view.removeFromSuperview()
            dotViews.removeValue(forKey: id)
        }

        for (id, label) in textLabels where !activeIDs.contains(id) {
            label.removeFromSuperview()
            textLabels.removeValue(forKey: id)
        }

        for person in people {
            let frame = visionRectToUIKitRect(person.normRect, in: bounds)

            let offsetFactor: CGFloat = 0.1
            let offset = frame.height * offsetFactor
            let rawY = frame.minY - offset

            let center = CGPoint(
                x: frame.midX,
                y: max(10, rawY)
            )

            let dotSize: CGFloat = 14

            let dotView: UIView
            if let existing = dotViews[person.id] {
                dotView = existing
            } else {
                let v = UIView()
                v.layer.cornerRadius = dotSize / 2
                v.layer.borderWidth = 2
                addSubview(v)
                dotViews[person.id] = v
                dotView = v
            }

            let color = uiColor(for: person.averageDepth)
            dotView.backgroundColor = color
            dotView.layer.borderColor = UIColor.white.cgColor

            dotView.frame = CGRect(
                x: center.x - dotSize / 2,
                y: center.y - dotSize / 2,
                width: dotSize,
                height: dotSize
            )

            let label: UILabel
            if let existing = textLabels[person.id] {
                label = existing
            } else {
                let l = UILabel()
                l.font = .monospacedSystemFont(ofSize: 12, weight: .bold)
                l.textColor = .white
                l.backgroundColor = UIColor.black.withAlphaComponent(0.65)
                l.layer.cornerRadius = 6
                l.layer.masksToBounds = true
                l.textAlignment = .center
                addSubview(l)
                textLabels[person.id] = l
                label = l
            }

            label.text = "P\(person.id)  \(String(format: "%.1fm", person.averageDepth))"
            label.sizeToFit()

            label.frame = CGRect(
                x: center.x - (label.bounds.width + 12) / 2,
                y: max(0, center.y - dotSize / 2 - 30),
                width: label.bounds.width + 12,
                height: 22
            )
        }
    }

    private func visionRectToUIKitRect(_ rect: CGRect, in bounds: CGRect) -> CGRect {
        CGRect(
            x: rect.minX * bounds.width,
            y: (1.0 - rect.maxY) * bounds.height,
            width: rect.width * bounds.width,
            height: rect.height * bounds.height
        )
    }

    private func uiColor(for metres: Float) -> UIColor {
        let t = CGFloat(min(max(metres / 8.0, 0), 1))
        return UIColor(hue: (1.0 - t) * 0.35, saturation: 1.0, brightness: 1.0, alpha: 1.0)
    }
}

// MARK: - Container

final class ARContainerView: UIView {
    let sceneView = ARSCNView()
    let overlayView = PersonBoundingBoxOverlayView()

    override init(frame: CGRect) {
        super.init(frame: frame)

        sceneView.frame = bounds
        sceneView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        addSubview(sceneView)

        overlayView.frame = bounds
        overlayView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        addSubview(overlayView)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}

// MARK: - AR View

struct ARCameraView: UIViewRepresentable {
    @ObservedObject var vm: ARViewModel

    func makeCoordinator() -> Coordinator {
        Coordinator(vm: vm)
    }

    func makeUIView(context: Context) -> ARContainerView {
        let container = ARContainerView()
        let scnView = container.sceneView

        scnView.scene = SCNScene()
        scnView.automaticallyUpdatesLighting = true

        scnView.session.delegate = context.coordinator
        context.coordinator.scnView = scnView
        context.coordinator.overlayView = container.overlayView

        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentationWithDepth) else {
            DispatchQueue.main.async {
                vm.statusMessage = "⚠️ Device does not support person segmentation with depth"
            }
            return container
        }

        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = [.personSegmentationWithDepth]
        scnView.session.run(config, options: [.resetTracking, .removeExistingAnchors])

        DispatchQueue.main.async {
            vm.isRunning = true
            vm.statusMessage = "Running"
        }

        return container
    }

    func updateUIView(_ uiView: ARContainerView, context: Context) {
        uiView.overlayView.updatePeople(vm.people)
    }

    final class Coordinator: NSObject, ARSessionDelegate {
        weak var scnView: ARSCNView?
        weak var overlayView: PersonBoundingBoxOverlayView?
        let vm: ARViewModel

        private let visionQueue = DispatchQueue(label: "vision.queue", qos: .userInitiated)
        private let humanRequest: VNDetectHumanRectanglesRequest
        private var isProcessing = false
        private var frameCounter = 0

        init(vm: ARViewModel) {
            self.vm = vm
            let request = VNDetectHumanRectanglesRequest()
            request.upperBodyOnly = false
            self.humanRequest = request
            super.init()
        }

        func session(_ session: ARSession, didUpdate frame: ARFrame) {
            frameCounter += 1

            guard !isProcessing else { return }

            // Process every 4th frame
            if frameCounter % 4 != 0 { return }

            guard frame.segmentationBuffer != nil,
                  frame.estimatedDepthData != nil else {
                DispatchQueue.main.async {
                    self.vm.statusMessage = "Waiting for seg+depth…"
                    self.vm.people = []
                    self.overlayView?.updatePeople([])
                }
                return
            }

            isProcessing = true

            visionQueue.async { [weak self] in
                guard let self else { return }
                defer { self.isProcessing = false }

                let handler = VNImageRequestHandler(
                    cvPixelBuffer: frame.capturedImage,
                    orientation: .right,
                    options: [:]
                )

                do {
                    try handler.perform([self.humanRequest])
                    let observations = (self.humanRequest.results as? [VNHumanObservation]) ?? []
                    let people = self.extractPeople(from: frame, observations: observations)

                    let msg = people.isEmpty
                        ? "No people detected"
                        : "\(people.count) person\(people.count == 1 ? "" : "s") detected"

                    DispatchQueue.main.async {
                        self.vm.people = people
                        self.vm.statusMessage = msg
                        self.overlayView?.updatePeople(people)
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.vm.statusMessage = "Vision error: \(error.localizedDescription)"
                        self.vm.people = []
                        self.overlayView?.updatePeople([])
                    }
                }
            }
        }

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

            var people: [PersonDepthInfo] = []

            for (index, obs) in observations.enumerated() {
                let bbox = obs.boundingBox

                let minX = max(0, Int(bbox.minX * CGFloat(segW)))
                let maxX = min(segW - 1, Int(bbox.maxX * CGFloat(segW)))

                let minYVision = max(0, Int(bbox.minY * CGFloat(segH)))
                let maxYVision = min(segH - 1, Int(bbox.maxY * CGFloat(segH)))

                let minY = segH - 1 - maxYVision
                let maxY = segH - 1 - minYVision

                guard minX <= maxX, minY <= maxY else { continue }

                let sampleEvery = 8
                var sampledDepths: [Float] = []
                sampledDepths.reserveCapacity(128)

                var validPixelCount = 0
                var minDepthSeen = Float.greatestFiniteMagnitude
                var maxDepthSeen: Float = 0

                for row in minY...maxY {
                    let rowBase = row * segBPR

                    for col in minX...maxX {
                        let segValue = segPtr[rowBase + col]
                        guard segValue != 0 else { continue }

                        let dc = min(depW - 1, Int(Float(col) * xScale))
                        let dr = min(depH - 1, Int(Float(row) * yScale))
                        let depth = depthPtr[dr * depthStride + dc]

                        guard depth.isFinite, depth > 0 else { continue }

                        validPixelCount += 1
                        minDepthSeen = min(minDepthSeen, depth)
                        maxDepthSeen = max(maxDepthSeen, depth)

                        if validPixelCount % sampleEvery == 0 {
                            sampledDepths.append(depth)
                        }
                    }
                }

                guard validPixelCount > 50, !sampledDepths.isEmpty else { continue }

                sampledDepths.sort()
                let median = sampledDepths[sampledDepths.count / 2]
                let rounded = Float(Int(median))
                people.append(
                    PersonDepthInfo(
                        id: index,
                        averageDepth: rounded,
                        minDepth: minDepthSeen,
                        maxDepth: maxDepthSeen,
                        pixelCount: validPixelCount,
                        normRect: bbox
                    )
                )
            }

            return people
        }

        func session(_ session: ARSession, didFailWithError error: Error) {
            DispatchQueue.main.async {
                self.vm.statusMessage = "Error: \(error.localizedDescription)"
            }
        }

        func sessionWasInterrupted(_ session: ARSession) {
            DispatchQueue.main.async {
                self.vm.statusMessage = "Interrupted"
            }
        }

        func sessionInterruptionEnded(_ session: ARSession) {
            DispatchQueue.main.async {
                self.vm.statusMessage = "Resuming…"
            }
        }
    }
}

// MARK: - UI

struct ContentView: View {
    @StateObject private var vm = ARViewModel()

    var body: some View {
        ZStack(alignment: .bottom) {
            ARCameraView(vm: vm)
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

    private var c: Color {
        let t = Double(min(person.averageDepth / 8.0, 1.0))
        return Color(hue: (1.0 - t) * 0.35, saturation: 1, brightness: 1)
    }

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(c)
                .frame(width: 10, height: 10)
                .shadow(color: c.opacity(0.8), radius: 4)

            VStack(alignment: .leading, spacing: 2) {
                Text("Person \(person.id)")
                    .font(.caption2.bold())
                    .foregroundColor(.white)

                Text(String(format: "%.2f m", person.averageDepth))
                    .font(.system(size: 15, weight: .bold, design: .monospaced))
                    .foregroundColor(c)

                Text(String(format: "%.1f – %.1f m", person.minDepth, person.maxDepth))
                    .font(.caption2)
                    .foregroundColor(.white.opacity(0.5))
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color.black.opacity(0.45))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(c.opacity(0.6), lineWidth: 1)
        )
    }
}
