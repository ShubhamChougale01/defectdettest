import SwiftUI
import CoreML
import Vision
import UIKit
import PhotosUI

struct ContentView: View {
    @State private var viewModel = DefectDetectionViewModel()
    @State private var isPresented = false
    
    let imageNames = [
        "defect_Wall_Hole_082.jpg",
        "cracked_window.jpg",
        "FLEOR_DAMAGES.png"
    ]
    
    var body: some View {
        NavigationView {
            Form {
                VStack(spacing: 20) {
                    // Image Display
                    if let uiImage = viewModel.selectedImage {
                        Image(uiImage: uiImage)
                            .resizable()
                            .scaledToFit()
                            .frame(height: 320)
                            .background(Color.gray.opacity(0.1))
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                    } else {
                        Rectangle()
                            .fill(Color.gray.opacity(0.2))
                            .frame(height: 320)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                            .overlay(Text("No Image Selected").foregroundColor(.gray))
                    }
                    
                    Text(viewModel.predictionLabel)
                        .font(.headline)
                        .multilineTextAlignment(.center)
                        .padding()
                    
                    // Image selection from fixed assets
                    Text("📸 Choose an image:")
                        .font(.subheadline)
                    
                    HStack {
                        ForEach(imageNames, id: \.self) { name in
                            Button(action: {
                                print("DEBUG: Button tapped for image: \(name)")
                                viewModel.loadImage(named: name)
                            }) {
                                Text(name)
                                    .font(.caption)
                                    .padding(8)
                                    .background(Color.blue.opacity(0.2))
                                    .clipShape(RoundedRectangle(cornerRadius: 8))
                            }
                        }
                    }
                    .padding(.horizontal)
                    
                    // Upload Image Button
                    Button(action: {
                        print("DEBUG: Upload Image button tapped")
                        isPresented = true
                    }) {
                        Text("Upload Image")
                            .font(.headline)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color.green)
                            .foregroundColor(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    .padding(.horizontal)
                    
                    // Reset Button
                    Button(action: {
                        print("DEBUG: Reset button tapped")
                        viewModel.reset()
                    }) {
                        Text("Reset")
                            .font(.headline)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color.red)
                            .foregroundColor(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    .padding(.horizontal)
                    
                    Spacer()
                }
                .padding()
                .navigationTitle("Defect Detection")
                .sheet(isPresented: $isPresented) {
                    ImagePicker(selectedImage: $viewModel.selectedImage)
                }
                .onChange(of: viewModel.selectedImage) { newImage in
                    if let image = newImage {
                        print("DEBUG: Selected image changed, running prediction")
                        viewModel.runPrediction(on: image)
                    }
                }
            }
        }
    }
}

class DefectDetectionViewModel: ObservableObject {
    @Published var predictionLabel: String = "🧠 Select an image to run prediction"
    @Published var selectedImage: UIImage?
    @Published var detectedObjects: [VNCoreMLFeatureValueObservation] = []
    
    private let inputSize = CGSize(width: 640, height: 640)
    private let anchors: [[Float]] = [
        [10, 13, 16, 30, 33, 23], // Small objects
        [30, 61, 62, 45, 59, 119], // Medium objects
        [116, 90, 156, 198, 373, 326] // Large objects
    ]
    private let strides: [Float] = [8, 16, 32] // Grid strides for different scales
    private var classLabels: [String] = []
    
    init() {
        loadModelMetadata()
    }
    
    private func loadModelMetadata() {
        guard let modelURL = Bundle.main.url(forResource: "defect_detect", withExtension: "mlmodelc") else {
            print("❌ Model not found")
            return
        }
        do {
            let model = try MLModel(contentsOf: modelURL, configuration: MLModelConfiguration())
            if let labels = model.modelDescription.classLabels as? [String] {
                classLabels = labels
            } else {
                classLabels = (0..<7).map { "Class \($0)" } // Fallback
            }
            print("Class labels from model: \(classLabels)")
            
            if let inputDesc = model.modelDescription.inputDescriptionsByName.first?.value {
                print("Model input: \(inputDesc)")
            }
        } catch {
            print("❌ Failed to load model metadata: \(error)")
        }
    }
    
    func loadImage(named name: String) {
        print("DEBUG: Attempting to load image: \(name)")
        if let img = loadAssetImage(named: name) {
            print("DEBUG: Successfully loaded image: \(name)")
            selectedImage = img
        } else {
            print("DEBUG: Failed to load image: \(name)")
            predictionLabel = "❌ Failed to load image \(name)"
        }
    }
    
    func reset() {
        selectedImage = nil
        predictionLabel = "🧠 Select an image to run prediction"
        detectedObjects = []
    }
    
    func runPrediction(on image: UIImage) {
        predictionLabel = "🧠 Running prediction..."
        
        let resizedImage = resizeImage(image, targetSize: inputSize)
        guard let ciImage = preprocessImage(resizedImage) else {
            predictionLabel = "❌ Could not convert image"
            return
        }
        
        guard let modelURL = Bundle.main.url(forResource: "defect_detect", withExtension: "mlmodelc") else {
            predictionLabel = "❌ Model not found"
            return
        }
        
        do {
            let config = MLModelConfiguration()
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            let vnModel = try VNCoreMLModel(for: model)
            
            let request = VNCoreMLRequest(model: vnModel) { request, error in
                if let error = error {
                    DispatchQueue.main.async {
                        self.predictionLabel = "❌ Prediction error: \(error.localizedDescription)"
                    }
                    print("Prediction error: \(error)")
                    return
                }
                
                if let results = request.results as? [VNCoreMLFeatureValueObservation],
                   let multiArray = results.first?.featureValue.multiArrayValue {
                    print("MultiArray shape: \(multiArray.shape)")
                    
                    let predictions = self.decodeYOLOOutput(multiArray, imageSize: image.size)
                    let filteredPredictions = self.nonMaxSuppression(predictions)
                    
                    DispatchQueue.main.async {
                        if filteredPredictions.isEmpty {
                            self.predictionLabel = "⚠️ No objects detected"
                        } else {
                            let top = filteredPredictions[0]
                            self.predictionLabel = String(format: "✅ %@ (%.2f%%)", top.label, min(top.confidence * 100, 100.0))
                            for p in filteredPredictions {
                                print("🔍 \(p.label): \(Int(min(p.confidence * 100, 100.0)))% at \(p.rect)")
                            }
                        }
                    }
                } else {
                    DispatchQueue.main.async {
                        self.predictionLabel = "⚠️ No valid prediction result"
                    }
                }
            }
            
            request.imageCropAndScaleOption = .centerCrop
            let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
            try handler.perform([request])
        } catch {
            predictionLabel = "❌ Prediction error: \(error.localizedDescription)"
            print("Model load or prediction error: \(error)")
        }
    }
    
    func resizeImage(_ image: UIImage, targetSize: CGSize) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }
    
    func preprocessImage(_ image: UIImage) -> CIImage? {
        guard let ciImage = CIImage(image: image) else { return nil }
        let filter = CIFilter(name: "CIColorMatrix")
        filter?.setValue(ciImage, forKey: kCIInputImageKey)
        filter?.setValue(CIVector(x: 2.0/255.0, y: 0, z: 0, w: -1), forKey: "inputRVector")
        filter?.setValue(CIVector(x: 0, y: 2.0/255.0, z: 0, w: -1), forKey: "inputGVector")
        filter?.setValue(CIVector(x: 0, y: 0, z: 2.0/255.0, w: -1), forKey: "inputBVector")
        filter?.setValue(CIVector(x: 0, y: 0, z: 0, w: 1), forKey: "inputAVector")
        return filter?.outputImage
    }
    
    func decodeYOLOOutput(_ multiArray: MLMultiArray, imageSize: CGSize) -> [(label: String, confidence: Float, rect: CGRect)] {
        var predictions: [(label: String, confidence: Float, rect: CGRect)] = []
        
        guard multiArray.shape.count == 3,
              let channels = multiArray.shape[1] as? Int,
              let boxes = multiArray.shape[2] as? Int else {
            print("❌ Unexpected multiArray shape: \(multiArray.shape)")
            return []
        }
        
        print("Output channels: \(channels)")
        let numClasses = channels - 5
        let threshold: Float = 0.7
        
        guard numClasses > 0, classLabels.count == numClasses else {
            print("❌ Class count mismatch: \(numClasses), expected \(classLabels.count)")
            return []
        }
        
        var maxScore: Float = .nan
        var minScore: Float = .nan
        var nonZeroCount: Int = 0
        for b in 0..<min(10, boxes) {
            let base = b * channels
            let classScores: [Float] = (0..<numClasses).map { multiArray[base + 5 + $0].floatValue }
            nonZeroCount += classScores.filter { $0 != 0 }.count
            if let max = classScores.max(), let min = classScores.min() {
                maxScore = maxScore.isNaN ? max : Swift.max(maxScore, max)
                minScore = minScore.isNaN ? min : Swift.min(minScore, min)
            }
        }
        print("Raw class score range: [\(minScore), \(maxScore)]")
        print("Non-zero class scores: \(nonZeroCount)")
        
        if nonZeroCount == 0 {
            print("⚠️ All class scores are zero. Model may be untrained.")
            return []
        }
        
        for b in 0..<boxes {
            let base = b * channels
            let rawX = multiArray[base + 0].floatValue
            let rawY = multiArray[base + 1].floatValue
            let rawWidth = multiArray[base + 2].floatValue
            let rawHeight = multiArray[base + 3].floatValue
            let rawObjectness = multiArray[base + 4].floatValue
            let objectness = sigmoid(rawObjectness)
            
            if objectness < threshold { continue }
            
            let classScores: [Float] = (0..<numClasses).map { multiArray[base + 5 + $0].floatValue }
            let classProbabilities = softmax(classScores)
            
            if let bestClassIndex = classProbabilities.firstIndex(of: classProbabilities.max() ?? 0),
               bestClassIndex < classLabels.count {
                let classConfidence = classProbabilities[bestClassIndex]
                let confidence = min(objectness * classConfidence, 1.0)
                
                if confidence < threshold { continue }
                
                let x = sigmoid(rawX)
                let y = sigmoid(rawY)
                let width = sigmoid(rawWidth)
                let height = sigmoid(rawHeight)
                
                let rect = CGRect(
                    x: CGFloat(x - width / 2) * imageSize.width,
                    y: CGFloat(y - height / 2) * imageSize.height,
                    width: CGFloat(width) * imageSize.width,
                    height: CGFloat(height) * imageSize.height
                )
                
                let label = classLabels[bestClassIndex]
                predictions.append((label: label, confidence: confidence, rect: rect))
            }
        }
        
        return predictions.sorted { $0.confidence > $1.confidence }
    }
    
    func nonMaxSuppression(_ predictions: [(label: String, confidence: Float, rect: CGRect)], iouThreshold: Float = 0.45) -> [(label: String, confidence: Float, rect: CGRect)] {
        var filtered: [(label: String, confidence: Float, rect: CGRect)] = []
        var remaining = predictions.sorted { $0.confidence > $1.confidence }
        
        while !remaining.isEmpty {
            let top = remaining.removeFirst()
            filtered.append(top)
            
            remaining = remaining.filter { pred in
                let iou = calculateIoU(top.rect, pred.rect)
                return iou < iouThreshold
            }
        }
        
        return filtered
    }
    
    func calculateIoU(_ rect1: CGRect, _ rect2: CGRect) -> Float {
        let intersection = rect1.intersection(rect2)
        let intersectionArea = intersection.width * intersection.height
        let unionArea = (rect1.width * rect1.height) + (rect2.width * rect2.height) - intersectionArea
        return Float(intersectionArea / max(unionArea, 1e-6))
    }
    
    func sigmoid(_ x: Float) -> Float {
        return 1 / (1 + exp(-x))
    }
    
    func softmax(_ scores: [Float]) -> [Float] {
        let maxScore = scores.max() ?? 0
        let expScores = scores.map { exp($0 - maxScore) }
        let sumExp = expScores.reduce(0, +)
        return sumExp == 0 ? scores.map { _ in 1.0 / Float(scores.count) } : expScores.map { $0 / sumExp }
    }
    
    func loadAssetImage(named name: String) -> UIImage? {
        print("DEBUG: Checking for image in bundle: \(name)")
        if let image = UIImage(named: name) {
            print("DEBUG: Loaded image from named asset: \(name)")
            return image
        }
        
        let components = name.split(separator: ".")
        guard components.count == 2,
              let path = Bundle.main.path(forResource: String(components[0]), ofType: String(components[1])),
              let image = UIImage(contentsOfFile: path) else {
            print("❌ Failed to load image from path: \(name)")
            return nil
        }
        print("DEBUG: Loaded image from file path: \(name)")
        return image
    }
}

// Image Picker
struct ImagePicker: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    
    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            picker.dismiss(animated: true)
            
            guard let provider = results.first?.itemProvider, provider.canLoadObject(ofClass: UIImage.self) else {
                print("DEBUG: No image selected in ImagePicker")
                return
            }
            
            provider.loadObject(ofClass: UIImage.self) { image, error in
                DispatchQueue.main.async {
                    if let uiImage = image as? UIImage {
                        print("DEBUG: Image selected from ImagePicker")
                        self.parent.selectedImage = uiImage
                    } else if let error = error {
                        print("DEBUG: Error loading image from ImagePicker: \(error)")
                    }
                }
            }
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .images
        config.selectionLimit = 1
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}
}

#Preview {
    ContentView()
}
