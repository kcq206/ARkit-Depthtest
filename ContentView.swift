import SwiftUI
import ARKit
import SceneKit
import Vision
import CoreVideo
import UIKit
import Combine

// MARK: - Shared State
class ARState: ObservableObject {
    @Published var personDetected: Bool = false
    @Published var detectedCount: Int = 0
    @Published var isPlaced: Bool = false
}

// MARK: - Root View
struct ContentView: View {
    @StateObject private var state = ARState()

    var body: some View {
        ZStack {
            ARSCNViewContainer(state: state)
                .edgesIgnoringSafeArea(.all)

            VStack {
                HStack {
                    Circle()
                        .fill(state.personDetected ? Color.green : Color.gray)
                        .frame(width: 10, height: 10)
                    Text(state.isPlaced
                         ? "Placed"
                         : (state.personDetected
                            ? "Tap to place (\(state.detectedCount) detected)"
                            : "No person detected"))
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.white)
                }
                .padding(8)
                .background(Color.black.opacity(0.5))
                .cornerRadius(8)
                .padding(.top, 60)
                Spacer()
            }
        }
    }
}

// MARK: - AR View Container
struct ARSCNViewContainer: UIViewRepresentable {
    let state: ARState

    func makeCoordinator() -> Coordinator { Coordinator(state: state) }

    func makeUIView(context: Context) -> ARSCNView {
        let arView = ARSCNView(frame: .zero)
        context.coordinator.arView = arView

        arView.delegate = context.coordinator
        arView.session.delegate = context.coordinator
        arView.scene = SCNScene()
        arView.automaticallyUpdatesLighting = true

        let config = ARWorldTrackingConfiguration()
        config.isAutoFocusEnabled = true
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            config.frameSemantics.insert(.smoothedSceneDepth)
        } else if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])

        let tap = UITapGestureRecognizer(target: context.coordinator,
                                         action: #selector(Coordinator.handleTap(_:)))
        arView.addGestureRecognizer(tap)

        arView.layer.addSublayer(context.coordinator.boxOverlayLayer)
        arView.layer.addSublayer(context.coordinator.headOverlayLayer)
        context.coordinator.configureOverlayLayers()

        return arView
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        context.coordinator.boxOverlayLayer.frame  = uiView.bounds
        context.coordinator.headOverlayLayer.frame = uiView.bounds
        CATransaction.commit()
    }

    // MARK: - Coordinator
    final class Coordinator: NSObject, ARSCNViewDelegate, ARSessionDelegate {
        weak var arView: ARSCNView?
        let state: ARState

        init(state: ARState) { self.state = state }

        private var isPlaced = false

        
        // head points, depth, and camera transform are all from the same snapshot.
        private var latestVisionFrame: ARFrame?

        private var latestHeadPoints: [CGPoint] = []
        private var smoothedHeadPoints: [CGPoint] = []
        private let headPointAlpha: CGFloat = 0.35

        private var latestBoundingBoxes: [CGRect] = []
        private var placedAnchor: ARAnchor?

        let boxOverlayLayer  = CAShapeLayer()
        let headOverlayLayer = CAShapeLayer()

        private let visionQueue = DispatchQueue(label: "vision.queue", qos: .userInteractive)
        private var visionBusy = false
        private var lastVisionTime: CFTimeInterval = 0
        private let visionHz: Double = 10.0

        func configureOverlayLayers() {
            boxOverlayLayer.fillColor   = UIColor.cyan.withAlphaComponent(0.08).cgColor
            boxOverlayLayer.strokeColor = UIColor.cyan.cgColor
            boxOverlayLayer.lineWidth   = 2.5
            boxOverlayLayer.zPosition   = 1

            headOverlayLayer.fillColor   = UIColor.red.withAlphaComponent(0.9).cgColor
            headOverlayLayer.strokeColor = UIColor.white.cgColor
            headOverlayLayer.lineWidth   = 1.5
            headOverlayLayer.zPosition   = 2
        }

        // MARK: - ARSessionDelegate
        func session(_ session: ARSession, didUpdate frame: ARFrame) {
            let t = CACurrentMediaTime()
            guard !visionBusy, t - lastVisionTime >= (1.0 / visionHz) else { return }
            lastVisionTime = t
            visionBusy = true

            // Capture the exact frame so Vision results and placement use the same snapshot.
            let visionFrame = frame
            visionQueue.async { [weak self] in
                self?.runVision(frame: visionFrame)
            }
        }

        // MARK: - Vision
        private func runVision(frame: ARFrame) {
            defer { DispatchQueue.main.async { self.visionBusy = false } }

            let poseReq = VNDetectHumanBodyPoseRequest()
            let rectReq = VNDetectHumanRectanglesRequest()
            rectReq.upperBodyOnly = false

            let handler = VNImageRequestHandler(cvPixelBuffer: frame.capturedImage,
                                                orientation: .right,
                                                options: [:])
            do { try handler.perform([poseReq, rectReq]) } catch {
                print("Vision error:", error); return
            }

            let poseObs = (poseReq.results as? [VNHumanBodyPoseObservation]) ?? []
            let rectObs = (rectReq.results as? [VNHumanObservation]) ?? []

            let headPoints: [CGPoint] = poseObs.compactMap { obs in
                for joint in [VNHumanBodyPoseObservation.JointName.nose, .leftEar, .rightEar, .neck] {
                    if let pt = try? obs.recognizedPoint(joint), pt.confidence > 0.3 {
                        return CGPoint(x: pt.x, y: pt.y)
                    }
                }
                return nil
            }

            let boundingBoxes = rectObs.map { $0.boundingBox }

            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                // Store the frame alongside results so tap placement uses the same snapshot.
                self.latestVisionFrame   = frame
                self.latestHeadPoints    = self.smoothAllHeadPoints(headPoints)
                self.latestBoundingBoxes = boundingBoxes
                self.state.personDetected = !headPoints.isEmpty
                self.state.detectedCount  = headPoints.count

                if let v = self.arView {
                    self.drawOverlays(headPoints: self.latestHeadPoints,
                                      boundingBoxes: boundingBoxes,
                                      in: v)
                }
            }
        }

        // MARK: - Draw overlays
        private func drawOverlays(headPoints: [CGPoint], boundingBoxes: [CGRect], in view: ARSCNView) {
            let vs = view.bounds.size

            CATransaction.begin()
            CATransaction.setDisableActions(true)
            boxOverlayLayer.frame  = view.bounds
            headOverlayLayer.frame = view.bounds
            CATransaction.commit()

            let boxPath = UIBezierPath()
            for bbox in boundingBoxes {
                let rect = CGRect(
                    x:      bbox.minX * vs.width,
                    y:      (1.0 - bbox.maxY) * vs.height,
                    width:  bbox.width  * vs.width,
                    height: bbox.height * vs.height
                )
                boxPath.append(UIBezierPath(roundedRect: rect, cornerRadius: 4))

                let cornerLen: CGFloat = min(rect.width, rect.height) * 0.18
                let inset: CGFloat = 1.5

                boxPath.move(to: CGPoint(x: rect.minX + inset, y: rect.minY + cornerLen))
                boxPath.addLine(to: CGPoint(x: rect.minX + inset, y: rect.minY + inset))
                boxPath.addLine(to: CGPoint(x: rect.minX + cornerLen, y: rect.minY + inset))

                boxPath.move(to: CGPoint(x: rect.maxX - cornerLen, y: rect.minY + inset))
                boxPath.addLine(to: CGPoint(x: rect.maxX - inset, y: rect.minY + inset))
                boxPath.addLine(to: CGPoint(x: rect.maxX - inset, y: rect.minY + cornerLen))

                boxPath.move(to: CGPoint(x: rect.minX + inset, y: rect.maxY - cornerLen))
                boxPath.addLine(to: CGPoint(x: rect.minX + inset, y: rect.maxY - inset))
                boxPath.addLine(to: CGPoint(x: rect.minX + cornerLen, y: rect.maxY - inset))

                boxPath.move(to: CGPoint(x: rect.maxX - cornerLen, y: rect.maxY - inset))
                boxPath.addLine(to: CGPoint(x: rect.maxX - inset, y: rect.maxY - inset))
                boxPath.addLine(to: CGPoint(x: rect.maxX - inset, y: rect.maxY - cornerLen))
            }
            boxOverlayLayer.path = boxPath.cgPath

            let headPath = UIBezierPath()
            for point in headPoints {
                let screenPt = CGPoint(x: point.x * vs.width, y: (1.0 - point.y) * vs.height)
                let radius: CGFloat = 8
                headPath.move(to: CGPoint(x: screenPt.x + radius, y: screenPt.y))
                headPath.addArc(withCenter: screenPt, radius: radius,
                                startAngle: 0, endAngle: .pi * 2, clockwise: true)
            }
            headOverlayLayer.path = headPath.cgPath
        }

        // MARK: - Smooth head points
        private func smoothAllHeadPoints(_ newPoints: [CGPoint]) -> [CGPoint] {
            guard !smoothedHeadPoints.isEmpty else {
                smoothedHeadPoints = newPoints; return newPoints
            }
            var result: [CGPoint] = []
            var usedIndices = Set<Int>()

            for new in newPoints {
                var bestIdx: Int?
                var bestDist = CGFloat.greatestFiniteMagnitude
                for (i, prev) in smoothedHeadPoints.enumerated() {
                    guard !usedIndices.contains(i) else { continue }
                    let dist = hypot(new.x - prev.x, new.y - prev.y)
                    if dist < bestDist { bestDist = dist; bestIdx = i }
                }
                if let idx = bestIdx, bestDist < 0.15 {
                    let prev = smoothedHeadPoints[idx]
                    result.append(CGPoint(x: prev.x + headPointAlpha * (new.x - prev.x),
                                          y: prev.y + headPointAlpha * (new.y - prev.y)))
                    usedIndices.insert(idx)
                } else {
                    result.append(new)
                }
            }
            smoothedHeadPoints = result
            return result
        }

        // MARK: - Tap to place
        @objc func handleTap(_ gesture: UITapGestureRecognizer) {
            guard !isPlaced,
                  let frame = latestVisionFrame,   // same frame Vision ran on
                  !latestHeadPoints.isEmpty,
                  let view = arView
            else { return }

            let tapPt    = gesture.location(in: view)
            let viewSize = view.bounds.size

            // Find the head point closest to the tap
            let targetHead = latestHeadPoints.min(by: { a, b in
                let aS = CGPoint(x: a.x * viewSize.width, y: (1.0 - a.y) * viewSize.height)
                let bS = CGPoint(x: b.x * viewSize.width, y: (1.0 - b.y) * viewSize.height)
                return hypot(aS.x - tapPt.x, aS.y - tapPt.y) < hypot(bS.x - tapPt.x, bS.y - tapPt.y)
            })!

            // Convert head point to raw landscape pixel coords.
            // Vision ran with .right orientation so its normalized x/y axes are rotated:
            //   raw pixel X = (1 - head.y) * imgW
            //   raw pixel Y = head.x       * imgH
            let imgW = CVPixelBufferGetWidth(frame.capturedImage)
            let imgH = CVPixelBufferGetHeight(frame.capturedImage)
            let u = (1.0 - Float(targetHead.y)) * Float(imgW - 1)
            let v = Float(targetHead.x) * Float(imgH - 1)

            // Sample depth at the head pixel
            let depthPB = frame.smoothedSceneDepth?.depthMap ?? frame.sceneDepth?.depthMap
            guard let depth = depthPB.flatMap({ sampleDepthMeters($0, uImage: u, vImage: v,
                                                                   imageW: imgW, imageH: imgH) }) else {
                print("FAILED — no depth available")
                return
            }

            // Unproject head pixel + depth to get the world position
            let worldPos = unprojectToWorld(u: u, v: v, depth: depth, camera: frame.camera)
            print("Placing at head world pos: \(worldPos)")
            placeAnchor(at: worldPos, view: view)
        }

        // MARK: - Placement
        private func placeAnchor(at position: SIMD3<Float>, view: ARSCNView) {
            var t = matrix_identity_float4x4
            t.columns.3 = SIMD4<Float>(position.x, position.y, position.z, 1.0)
            let anchor = ARAnchor(transform: t)
            placedAnchor = anchor
            view.session.add(anchor: anchor)
            isPlaced = true
            print("Anchor placed at: \(t.columns.3)")
            DispatchQueue.main.async { self.state.isPlaced = true }
        }

        // MARK: - Render anchor
        func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
            guard anchor.identifier == placedAnchor?.identifier else { return }

            let plane = SCNPlane(width: 0.3, height: 0.3)
            let mat   = SCNMaterial()
            mat.diffuse.contents     = UIImage(named: "placement_image")
            mat.isDoubleSided        = true
            mat.readsFromDepthBuffer = true
            mat.writesToDepthBuffer  = true
            plane.materials = [mat]

            let planeNode = SCNNode(geometry: plane)
            planeNode.name = "PlacedPlane"
            let billboard = SCNBillboardConstraint()
            billboard.freeAxes = .all
            planeNode.constraints = [billboard]
            node.addChildNode(planeNode)
            print("Plane attached to anchor node")
        }

        // MARK: - Depth sampling (5×5 median)
        private func sampleDepthMeters(_ depthPB: CVPixelBuffer,
                                       uImage: Float, vImage: Float,
                                       imageW: Int, imageH: Int) -> Float? {
            CVPixelBufferLockBaseAddress(depthPB, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(depthPB, .readOnly) }

            let dW = CVPixelBufferGetWidth(depthPB)
            let dH = CVPixelBufferGetHeight(depthPB)
            let x  = Int(round((uImage / Float(imageW)) * Float(dW - 1)))
            let y  = Int(round((vImage / Float(imageH)) * Float(dH - 1)))

            guard let base = CVPixelBufferGetBaseAddress(depthPB) else { return nil }
            let ptr     = base.assumingMemoryBound(to: Float32.self)
            let strideF = CVPixelBufferGetBytesPerRow(depthPB) / MemoryLayout<Float32>.stride

            func depthAt(_ xx: Int, _ yy: Int) -> Float? {
                guard xx >= 0, xx < dW, yy >= 0, yy < dH else { return nil }
                let z = Float(ptr[(yy * strideF) + xx])
                return (z.isFinite && z >= 0.05 && z <= 20.0) ? z : nil
            }

            var samples = [Float]()
            samples.reserveCapacity(25)
            for dy in -2...2 { for dx in -2...2 { if let z = depthAt(x+dx, y+dy) { samples.append(z) } } }
            guard !samples.isEmpty else { return nil }
            samples.sort()
            return samples[samples.count / 2]
        }

        // MARK: - Unproject
        private func unprojectToWorld(u: Float, v: Float,
                                      depth: Float, camera: ARCamera) -> SIMD3<Float> {
            let K  = camera.intrinsics
            let fx = K.columns.0.x, fy = K.columns.1.y
            let cx = K.columns.2.x, cy = K.columns.2.y
            let pCam   = SIMD4<Float>((u - cx) / fx * depth, -(v - cy) / fy * depth, -depth, 1.0)
            let pWorld = simd_mul(camera.transform, pCam)
            return SIMD3<Float>(pWorld.x, pWorld.y, pWorld.z)
        }
    }
}
