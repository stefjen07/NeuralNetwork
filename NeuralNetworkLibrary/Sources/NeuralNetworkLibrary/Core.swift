//
//  Core.swift
//  NeuralNetwork
//
//  Created by Евгений on 15.04.2021.
//

import Foundation

func outNeuron(_ neuron: Neuron, input: [Float]) -> Float {
    var out = neuron.bias
    let weightsCount = neuron.weights.count
    input.withUnsafeBufferPointer { inputPtr in
        neuron.weights.withUnsafeBufferPointer { weightsPtr in
            DispatchQueue.concurrentPerform(iterations: weightsCount, execute: { i in
                out += weightsPtr[i] * inputPtr[i]
            })
        }
    }
    return out
}

public enum DataSizeType: Int, Codable {
    case oneD = 1
    case twoD
    case threeD
}

public struct DataSize: Codable {
    var type: DataSizeType
    var width: Int
    var height: Int?
    var depth: Int?
    
    public init(width: Int) {
        type = .oneD
        self.width = width
    }
    
    public init(width: Int, height: Int) {
        type = .twoD
        self.width = width
        self.height = height
    }
    
    public init(width: Int, height: Int, depth: Int) {
        type = .threeD
        self.width = width
        self.height = height
        self.depth = depth
    }
    
}

public struct DataPiece: Codable, Equatable {
    public static func == (lhs: DataPiece, rhs: DataPiece) -> Bool {
        return lhs.body == rhs.body
    }
    
    public var size: DataSize
    public var body: [Float]
    
    func get(x: Int) -> Float {
        return body[x]
    }
    
    func get(x: Int, y: Int) -> Float {
        return body[x+y*size.width]
    }
    
    func get(x: Int, y: Int, z: Int) -> Float {
        return body[z+(x+y*size.width)*size.depth!]
    }
    
    public init(size: DataSize, body: [Float]) {
        var flatSize = size.width
        if let height = size.height {
            flatSize *= height
        }
        if let depth = size.depth {
            flatSize *= depth
        }
        if flatSize != body.count {
            fatalError("DataPiece body does not conform to DataSize.")
        }
        self.size = size
        self.body = body
    }
    
    public init(image: CGImage) {
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let width = 32
        let height = 32
        let bitsPerComponent = image.bitsPerComponent
        let bytesPerRow = image.bytesPerRow
        let totalBytes = height * bytesPerRow
        let buffer = Array(repeating: UInt8.zero, count: totalBytes)
        let mutablePointer = UnsafeMutablePointer<UInt8>(mutating: buffer)
        let contextRef = CGContext(data: mutablePointer, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: 0)!
        contextRef.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        let bufferPointer = UnsafeBufferPointer<UInt8>(start: mutablePointer, count: totalBytes)
        self.size = .init(width: width, height: height)
        let pixelValues = Array<UInt8>(bufferPointer)
        self.body = pixelValues.map { v in
            return Float(v)/Float(UInt8.max)
        }
    }
}

public struct DataItem: Codable {
    var input: DataPiece
    var output: DataPiece
    
    public init(input: DataPiece, output: DataPiece) {
        self.input = input
        self.output = output
    }
    
    public init(input: [Float], inputSize: DataSize, output: [Float], outputSize: DataSize) {
        self.input = DataPiece(size: inputSize, body: input)
        self.output = DataPiece(size: outputSize, body: output)
    }
}

public struct Dataset: Codable {
    public var items: [DataItem]
    
    public func save(to url: URL) {
        let encoder = JSONEncoder()
        guard let encoded = try? encoder.encode(self) else {
            print("Unable to encode model.")
            return
        }
        do {
            try encoded.write(to: url)
        } catch {
            print("Unable to write model to disk.")
        }
    }
    
    public init(from url: URL) {
        let decoder = JSONDecoder()
        guard let data = try? Data(contentsOf: url) else {
            fatalError("Unable to get data from Dataset file.")
        }
        guard let decoded = try? decoder.decode(Dataset.self, from: data) else {
            fatalError("Unable to decode data from Dataset file.")
        }
        self.items = decoded.items
    }
    
    public init(items: [DataItem]) {
        self.items = items
    }
    
    public init(folderPath: String) {
        self.items = []
        let manager = FileManager.default
        var inputs = [DataPiece]()
        var outputs = [Int]()
        var classCount = 0
        do {
            for content in try manager.contentsOfDirectory(atPath: folderPath) {
                let isDirectory = UnsafeMutablePointer<ObjCBool>.allocate(capacity: 1)
                let path = folderPath+"/"+content
                if manager.fileExists(atPath: path, isDirectory: UnsafeMutablePointer<ObjCBool>(isDirectory)) {
                    if isDirectory.pointee.boolValue {
                        for file in try manager.contentsOfDirectory(atPath: path) {
                            let isDirectory = UnsafeMutablePointer<ObjCBool>.allocate(capacity: 1)
                            let path = path+"/"+file
                            if manager.fileExists(atPath: path, isDirectory: UnsafeMutablePointer<ObjCBool>(isDirectory)) {
                                if !isDirectory.pointee.boolValue {
                                    let url = URL(fileURLWithPath: path)
                                    let splits = file.split(separator: ".")
                                    if splits.count < 2 {
                                        continue
                                    }
                                    if splits.last  == "png" {
                                        let data = try Data(contentsOf: url)
                                        guard let provider = CGDataProvider(data: NSData(data: data)) else { fatalError() }
                                        guard let image = CGImage(pngDataProviderSource: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { fatalError() }
                                        inputs.append(.init(image: image))
                                        outputs.append(classCount)
                                    }
                                }
                            }
                        }
                        classCount += 1
                    }
                }
            }
        } catch {
            fatalError(error.localizedDescription)
        }
        for i in 0..<inputs.count {
            items.append(.init(input: inputs[i], output: classifierOutput(classes: classCount, correct: outputs[i])))
        }
    }
}

public class NeuralNetwork: Codable {
    public var layers: [Layer] = []
    public var learningRate = Float(0.05)
    public var epochs = 30
    public var batchSize = 16
    var dropoutEnabled = true
    
    private enum CodingKeys: String, CodingKey {
        case layers
        case learningRate
        case epochs
        case batchSize
    }
    
    public func printSummary() {
        for rawLayer in layers {
            switch rawLayer {
            case _ as Flatten:
                print("Flatten layer")
            case let layer as Convolutional2D:
                print("Convolutional 2D layer: \(layer.filters.count) filters")
            case let layer as Dense:
                print("Dense layer: \(layer.neurons.count) neurons")
            case let layer as Dropout:
                print("Dropout layer: \(layer.neurons.count) neurons, \(layer.probability) probability")
            default:
                break
            }
        }
    }
    
    public func encode(to encoder: Encoder) throws {
        let wrappers = layers.map { LayerWrapper($0) }
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(wrappers, forKey: .layers)
        try container.encode(learningRate, forKey: .learningRate)
        try container.encode(epochs, forKey: .epochs)
        try container.encode(batchSize, forKey: .batchSize)
    }
    
    public init(fileName: String) {
        let decoder = JSONDecoder()
        let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(fileName)
        guard let data = try? Data(contentsOf: url) else {
            print("Unable to read model from file.")
            return
        }
        guard let decoded = try? decoder.decode(NeuralNetwork.self, from: data) else {
            print("Unable to decode model.")
            return
        }
        self.layers = decoded.layers
        self.learningRate = decoded.learningRate
        self.epochs = decoded.epochs
        self.batchSize = decoded.batchSize
    }
    
    public init() {
        
    }
    
    public required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let wrappers = try container.decode([LayerWrapper].self, forKey: .layers)
        self.layers = wrappers.map { $0.layer }
        self.learningRate = try container.decode(Float.self, forKey: .learningRate)
        self.epochs = try container.decode(Int.self, forKey: .epochs)
        self.batchSize = try container.decode(Int.self, forKey: .batchSize)
    }
    
    public func saveModel(fileName: String) {
        let encoder = JSONEncoder()
        guard let encoded = try? encoder.encode(self) else {
            print("Unable to encode model.")
            return
        }
        let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(fileName)
        do {
            try encoded.write(to: url)
        } catch {
            print("Unable to write model to disk.")
        }
    }
    
    public func train(set: Dataset) {
        dropoutEnabled = true
        for epoch in 0..<epochs {
            var shuffledSet = set.items.shuffled()
            var error = Float.zero
            while !shuffledSet.isEmpty {
                let batch = shuffledSet.prefix(batchSize)
                for item in batch {
                    let predictions = forward(networkInput: item.input)
                    for i in 0..<item.output.body.count {
                        error+=pow(item.output.body[i]-predictions.body[i], 2)/2
                    }
                    backward(expected: item.output)
                    deltaWeights(row: item.input)
                }
                for layer in layers {
                    layer.updateWeights()
                }
                shuffledSet.removeFirst(min(batchSize,shuffledSet.count))
            }
            print("Epoch \(epoch+1), error \(error).")
        }
    }
    
    public func predict(input: DataPiece) -> Int {
        dropoutEnabled = false
        let output = forward(networkInput: input)
        #warning("Add more output activation functions")
        var maxi = 0
        for i in 1..<output.body.count {
            if(output.body[i]>output.body[maxi]) {
                maxi = i
            }
        }
        return maxi
    }
    
    func deltaWeights(row: DataPiece) {
        var input = row
        for i in 0..<layers.count {
            input = layers[i].deltaWeights(input: input, learningRate: learningRate)
        }
    }
    
    func forward(networkInput: DataPiece) -> DataPiece {
        var input = networkInput
        for i in 0..<layers.count {
            input = layers[i].forward(input: input, dropoutEnabled: dropoutEnabled)
        }
        return input
    }
    
    func backward(expected: DataPiece) {
        var input = expected
        var previous: Layer? = nil
        for i in (0..<layers.count).reversed() {
            input = layers[i].backward(input: input, previous: previous)
            if !(layers[i] is Dropout) {
                previous = layers[i]
            }
        }
    }
}

struct Neuron: Codable {
    var weights: [Float]
    var weightsDelta: [Float]
    var bias: Float
    var delta: Float
}

public func classifierOutput(classes: Int, correct: Int) -> DataPiece {
    if correct>=classes {
        fatalError("Correct class must be less than classes number.")
    }
    var output = Array(repeating: Float.zero, count: classes)
    output[correct] = 1.0
    return DataPiece(size: .init(width: classes), body: output)
}

func matrixSum(matrix: [[Float]]) -> Float {
    var output = Float.zero
    matrix.withUnsafeBufferPointer { matrixPtr in
        DispatchQueue.concurrentPerform(iterations: matrixPtr.count, execute: { i in
            matrixPtr[i].withUnsafeBufferPointer { matrixRowPtr in
                DispatchQueue.concurrentPerform(iterations: matrixRowPtr.count, execute: { j in
                    output += matrixRowPtr[j]
                })
            }
        })
    }
    return output
}
