//
//  Core.swift
//  NeuralNetwork
//
//  Created by Евгений on 15.04.2021.
//

import Foundation

func outNeuron(_ neuron: Neuron, input: [Float]) -> Float {
    var out = neuron.bias
    for i in 0..<neuron.weights.count {
        out += neuron.weights[i] * input[i]
    }
    return out
}

enum DataSizeType {
    case oneD
    case twoD
    case threeD
}

struct DataSize {
    var type: DataSizeType
    var width: Int
    var height: Int?
    var depth: Int?
    
    init(width: Int) {
        type = .oneD
        self.width = width
    }
    
    init(width: Int, height: Int) {
        type = .twoD
        self.width = width
        self.height = height
    }
    
    init(width: Int, height: Int, depth: Int) {
        type = .threeD
        self.width = width
        self.height = height
        self.depth = depth
    }
    
}

struct DataPiece {
    var size: DataSize
    var body: [Float]
    
    func get(x: Int) -> Float {
        return body[x]
    }
    
    func get(x: Int, y: Int) -> Float {
        return body[x+y*size.width]
    }
    
    func get(x: Int, y: Int, z: Int) -> Float {
        return body[z+(x+y*size.width)*size.depth!]
    }
}

struct DataItem {
    var input: DataPiece
    var output: DataPiece
    
    init(input: DataPiece, output: DataPiece) {
        self.input = input
        self.output = output
    }
    
    init(input: [Float], inputSize: DataSize, output: [Float], outputSize: DataSize) {
        self.input = DataPiece(size: inputSize, body: input)
        self.output = DataPiece(size: outputSize, body: output)
    }
}

struct Dataset {
    var items: [DataItem]
}

class NeuralNetwork: Codable {
    var layers: [Layer] = []
    var learningRate = Float(0.05)
    var epochs = 30
    var batchSize = 16
    var dropoutEnabled = true
    
    private enum CodingKeys: String, CodingKey {
        case layers
        case learningRate
        case epochs
        case batchSize
    }
    
    func printSummary() {
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
    
    func encode(to encoder: Encoder) throws {
        let wrappers = layers.map { LayerWrapper($0) }
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(wrappers, forKey: .layers)
        try container.encode(learningRate, forKey: .learningRate)
        try container.encode(epochs, forKey: .epochs)
        try container.encode(batchSize, forKey: .batchSize)
    }
    
    init(fileName: String) {
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
    
    init() {
        
    }
    
    required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let wrappers = try container.decode([LayerWrapper].self, forKey: .layers)
        self.layers = wrappers.map { $0.layer }
        self.learningRate = try container.decode(Float.self, forKey: .learningRate)
        self.epochs = try container.decode(Int.self, forKey: .epochs)
        self.batchSize = try container.decode(Int.self, forKey: .batchSize)
    }
    
    func saveModel(fileName: String) {
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
    
    func train(set: Dataset) {
        dropoutEnabled = true
        for epoch in 0..<epochs {
            let batch = set.items.shuffled().prefix(batchSize)
            var error = Float.zero
            for item in batch {
                let predictions = forward(networkInput: item.input)
                for i in 0..<item.output.body.count {
                    error+=pow(item.output.body[i]-predictions.body[i], 2)
                }
                backward(expected: item.output)
                updateWeights(row: item.input)
            }
            print("Epoch \(epoch+1), error \(error).")
            learningRate -= pow(learningRate, 6)
        }
    }
    
    func predict(input: DataPiece) -> Int {
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
    
    func updateWeights(row: DataPiece) {
        var input = row
        for layer in layers {
            input = layer.updateWeights(input: input, learningRate: learningRate)
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
        for i in (0..<layers.count).reversed() {
            if i<layers.count-1 {
                if layers[i+1] is Flatten {
                    continue
                }
            }
            let layer = layers[i]
            input = layer.backward(input: input, previous: i<layers.count-1 ? layers[i+1] : nil)
        }
    }
}

struct Neuron: Codable {
    var weights: [Float]
    var bias: Float
    var delta: Float
    var output: Float
}

func derivative(output: Float) -> Float {
    return output*(1.0-output)
}

func classifierOutput(classes: Int, correct: Int) -> DataPiece {
    if correct>classes {
        fatalError("Correct class must be less or equal classes number.")
    }
    var output = Array(repeating: Float.zero, count: classes)
    output[correct-1] = 1.0
    return DataPiece(size: .init(width: classes), body: output)
}

func matrixSum(matrix: [[Float]]) -> Float {
    var output = Float.zero
    for i in 0..<matrix.count {
        for j in 0..<matrix[i].count {
            output += matrix[i][j]
        }
    }
    return output
}
