import SwiftUI
import ARKit
import SceneKit
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
            self.vm.isRunning    = true
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

        init(vm: ARViewModel) { self.vm = vm }

        // -----------------------------------------------------------------
        // MARK: ARSessionDelegate
        // -----------------------------------------------------------------
        func session(_ session: ARSession, didUpdate frame: ARFrame) {
            // Throttle to every 3rd frame
            frameCounter = (frameCounter + 1) % 3
            guard frameCounter == 0 else { return }

            // Run the heavy pixel loop on a background queue
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self else { return }
                let result = self.extractPeople(from: frame)
                let msg    = result.isEmpty
                    ? "No people detected"
                    : "\(result.count) person\(result.count == 1 ? "" : "s") detected"
                DispatchQueue.main.async {
                    self.vm.people        = result
                    self.vm.statusMessage = msg
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
        // MARK: Core extraction – reads segmentationBuffer + estimatedDepthData
        // -----------------------------------------------------------------
        private func extractPeople(from frame: ARFrame) -> [PersonDepthInfo] {

            guard let segBuffer   = frame.segmentationBuffer,
                  let depthBuffer = frame.estimatedDepthData else {
                DispatchQueue.main.async { self.vm.statusMessage = "Waiting for seg+depth…" }
                return []
            }

            CVPixelBufferLockBaseAddress(segBuffer,   .readOnly)
            CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
            defer {
                CVPixelBufferUnlockBaseAddress(segBuffer,   .readOnly)
                CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)
            }

            guard let segBase   = CVPixelBufferGetBaseAddress(segBuffer),
                  let depthBase = CVPixelBufferGetBaseAddress(depthBuffer) else { return [] }

            // Dimensions & strides
            let segW        = CVPixelBufferGetWidth(segBuffer)
            let segH        = CVPixelBufferGetHeight(segBuffer)
            let depW        = CVPixelBufferGetWidth(depthBuffer)
            let depH        = CVPixelBufferGetHeight(depthBuffer)
            let segBPR      = CVPixelBufferGetBytesPerRow(segBuffer)
            let depBPR      = CVPixelBufferGetBytesPerRow(depthBuffer)
            let depthStride = depBPR / MemoryLayout<Float>.size

            //create scale to match resolution
            let xScale = Double(depW) / Double(segW)
            let yScale = Double(depH) / Double(segH)

            let segPtr   = segBase.assumingMemoryBound(to: UInt8.self)
            let depthPtr = depthBase.assumingMemoryBound(to: Float.self)

            struct Accum {
                var depthSum: Double = 0
                var depthMin: Float  =  .infinity
                var depthMax: Float  = -.infinity
                var count    = 0
                var minX = Int.max, minY = Int.max
                var maxX = Int.min, maxY = Int.min
            }
            var accums = [UInt8: Accum]()

            for row in 0 ..< segH {
                let rowBase = row * segBPR
                for col in 0 ..< segW {
                    let label = segPtr[rowBase + col]
                    guard label > 0 else { continue }

                    let dc    = min(Int(Double(col) * xScale), depW - 1)
                    let dr    = min(Int(Double(row) * yScale), depH - 1)
                    let depth = depthPtr[dr * depthStride + dc]
                    guard depth.isFinite, depth > 0 else { continue }

                    var a = accums[label, default: Accum()]
                    a.depthSum += Double(depth)
                    a.depthMin  = min(a.depthMin, depth)
                    a.depthMax  = max(a.depthMax, depth)
                    a.count    += 1
                    if col < a.minX { a.minX = col }
                    if col > a.maxX { a.maxX = col }
                    if row < a.minY { a.minY = row }
                    if row > a.maxY { a.maxY = row }
                    accums[label] = a
                }
            }

            return accums
                .sorted { $0.key < $1.key }
                .compactMap { label, a in
                    guard a.count > 50 else { return nil }
                    let normRect = CGRect(
                        x:      CGFloat(a.minX) / CGFloat(segW),
                        y:      CGFloat(a.minY) / CGFloat(segH),
                        width:  CGFloat(a.maxX - a.minX) / CGFloat(segW),
                        height: CGFloat(a.maxY - a.minY) / CGFloat(segH)
                    )
                    return PersonDepthInfo(
                        id:           Int(label),
                        averageDepth: Float(a.depthSum / Double(a.count)),
                        minDepth:     a.depthMin,
                        maxDepth:     a.depthMax,
                        pixelCount:   a.count,
                        normRect:     normRect
                    )
                }
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
                let r  = person.normRect
                let cx = r.midX  * geo.size.width
                // Place dot just above the top edge of the bounding box
                let cy = (r.minY * geo.size.height) - 40
                let c  = depthColor(person.averageDepth)

                ZStack {
                    // Outer glow ring
                    Circle().fill(c.opacity(0.25)).frame(width: 48, height: 48)
                    // Solid core
                    Circle().fill(c).frame(width: 22, height: 22)
                        .shadow(color: c.opacity(0.9), radius: 8)
                    // Distance label below the dot
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
