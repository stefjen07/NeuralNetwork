//
//  Layers.swift
//  NeuralNetwork
//
//  Created by Евгений on 15.04.2021.
//

import Foundation

struct LayerWrapper: Codable {
    let layer: Layer
    
    private enum CodingKeys: String, CodingKey {
        case base
        case payload
    }

    private enum Base: Int, Codable {
        case dense = 0
        case dropout
        case conv2d
        case flatten
    }
    
    init(_ layer: Layer) {
        self.layer = layer
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch layer {
        case let payload as Dense:
            try container.encode(Base.dense, forKey: .base)
            try container.encode(payload, forKey: .payload)
        case let payload as Dropout:
            try container.encode(Base.dropout, forKey: .base)
            try container.encode(payload, forKey: .payload)
        case let payload as Convolutional2D:
            try container.encode(Base.conv2d, forKey: .base)
            try container.encode(payload, forKey: .payload)
        case let payload as Flatten:
            try container.encode(Base.flatten, forKey: .base)
            try container.encode(payload, forKey: .payload)
        default:
            fatalError()
        }
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let base = try container.decode(Base.self, forKey: .base)
        
        switch base {
        case .dense:
            self.layer = try container.decode(Dense.self, forKey: .payload)
        case .dropout:
            self.layer = try container.decode(Dropout.self, forKey: .payload)
        case .conv2d:
            self.layer = try container.decode(Convolutional2D.self, forKey: .payload)
        case .flatten:
            self.layer = try container.decode(Flatten.self, forKey: .payload)
        }
    }

}

class Layer: Codable {
    var neurons: [Neuron] = []
    var function: ActivationFunction
    
    private enum CodingKeys: String, CodingKey {
        case neurons
        case function
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(function.rawValue, forKey: .function)
        try container.encode(neurons, forKey: .neurons)
    }
    
    required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let activationRaw = try container.decode(Int.self, forKey: .function)
        function = getActivationFunction(rawValue: activationRaw)
        neurons = try container.decode([Neuron].self, forKey: .neurons)
    }
    
    func forward(input: DataPiece, dropoutEnabled: Bool) -> DataPiece {
        return input
    }
    
    func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        return input
    }
    
    init(function: ActivationFunction) {
        self.function = function
    }
}

struct Filter: Codable {
    var kernel: [[Float]]
    
    func apply(to piece: [[Float]]) -> [[Float]] {
        if piece.count != kernel.count {
            fatalError("Piece size must be equal kernel size.")
        }
        var output = piece
        for i in 0..<piece.count {
            if piece[i].count != kernel[i].count {
                fatalError("Piece size must be equal kernel size.")
            }
            for j in 0..<piece[i].count {
                output[i][j] *= kernel[i][j]
            }
        }
        return output
    }
    
    init(kernelSize: Int) {
        kernel = []
        for _ in 0..<kernelSize {
            var row = [Float]()
            for _ in 0..<kernelSize {
                row.append(Float.random(in: -1...1))
            }
            kernel.append(row)
        }
    }
}

class Flatten: Layer {
    
    init() {
        super.init(function: Plain())
    }
    
    required init(from decoder: Decoder) throws {
        try super.init(from: decoder)
    }
    
    override func forward(input: DataPiece, dropoutEnabled: Bool) -> DataPiece {
        var newWidth = input.size.width
        if let height = input.size.height {
            newWidth *= height
        }
        if let depth = input.size.depth {
            newWidth *= depth
        }
        var output = input
        output.size = .init(width: newWidth)
        return output
    }
}

class Convolutional2D: Layer {
    var filters: [Filter]
    var kernelSize: Int
    var stride: Int
    
    private enum CodingKeys: String, CodingKey {
        case filters
        case kernelSize
        case stride
    }
    
    override func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(filters, forKey: .filters)
        try container.encode(kernelSize, forKey: .kernelSize)
        try container.encode(stride, forKey: .stride)
        try super.encode(to: encoder)
    }
    
    required init(from decoder: Decoder) throws {
        fatalError("Convolutional 2D not implemented yet.")
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.filters = try container.decode([Filter].self, forKey: .filters)
        self.kernelSize = try container.decode(Int.self, forKey: .kernelSize)
        self.stride = try container.decode(Int.self, forKey: .stride)
        try super.init(from: decoder)
    }
    
    init(filters: Int, kernelSize: Int, stride: Int, functionRaw: ActivationFunctionRaw) {
        fatalError("Convolutional 2D not implemented yet.")
        let function = getActivationFunction(rawValue: functionRaw.rawValue)
        self.filters = []
        self.kernelSize = kernelSize
        self.stride = stride
        for _ in 0..<filters {
            self.filters.append(Filter(kernelSize: kernelSize))
        }
        super.init(function: function)
    }
    
    override func forward(input: DataPiece, dropoutEnabled: Bool) -> DataPiece {
        let inputSize = DataSize(width: input.size.width, height: input.size.height ?? input.size.width)
        let outputSize = DataSize(width: (inputSize.width - kernelSize) / stride + 1, height: (input.size.height! - kernelSize) / stride + 1)
        var output = Array(repeating: Array(repeating: Float.zero, count: outputSize.width), count: outputSize.height!)
        for filter in filters {
            var tempY = 0, outY = 0
            while tempY + kernelSize <= inputSize.height! {
                var tempX = 0, outX = 0
                while tempX + kernelSize <= inputSize.width {
                    var piece = [[Float]]()
                    for y in tempY ..< tempY + kernelSize {
                        var column = [Float]()
                        for x in tempX ..< tempX + kernelSize {
                            column.append(input.get(x: x, y: y))
                        }
                        piece.append(column)
                    }
                    output[outY][outX] = function.activation(input: matrixSum(matrix: filter.apply(to: piece)))
                    tempX += stride
                    outX += 1
                }
                tempY += stride
                outY += 1
            }
        }
        var flat = [Float]()
        for i in output {
            flat.append(contentsOf: i)
        }
        return DataPiece(size: outputSize, body: flat)
    }
    
    override func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        let inputSize = input.size
        for filter in filters {
            var tempY = 0, outY = 0
            while tempY + kernelSize <= inputSize.height! {
                var tempX = 0, outX = 0
                while tempX + kernelSize <= inputSize.width {
                    var piece = [[Float]]()
                    for y in tempY ..< tempY + kernelSize {
                        var column = [Float]()
                        for x in tempX ..< tempX + kernelSize {
                            column.append(input.get(x: x, y: y))
                        }
                        piece.append(column)
                    }
                    tempX += stride
                    outX += 1
                }
            }
            tempY += stride
            outY += 1
        }
        #warning("Change return")
        return input
    }
    
}

class Dense: Layer {
    
    override func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        var errors = [Float]()
        if let previous = previous {
            for j in 0..<neurons.count {
                var error = Float.zero
                if !(previous is Dropout) {
                    for neuron in previous.neurons {
                        error += neuron.weights[j]*neuron.delta
                    }
                }
                errors.append(error)
            }
        } else {
            for j in 0..<neurons.count {
                errors.append(input.body[j]-neurons[j].output)
            }
        }
        for j in 0..<neurons.count {
            neurons[j].delta = errors[j] * transferDerivative(output: neurons[j].output)
        }
        var output = [Float]()
        for neuron in neurons {
            output.append(neuron.output)
        }
        return DataPiece(size: .init(width: output.count), body: output)
    }
    
    init(inputSize: Int, neuronsCount: Int, functionRaw: ActivationFunctionRaw) {
        let function = getActivationFunction(rawValue: functionRaw.rawValue)
        
        super.init(function: function)
        
        for _ in 0..<neuronsCount {
            var weights = [Float]()
            for _ in 0..<inputSize {
                weights.append(Float.random(in: -1.0 ... 1.0))
            }
            neurons.append(Neuron(weights: weights, bias: 0.0, delta: 0.0, output: 0.0))
        }
    }
    
    required init(from decoder: Decoder) throws {
        try super.init(from: decoder)
    }
    
    override func forward(input: DataPiece, dropoutEnabled: Bool) -> DataPiece {
        var newInput = [Float]()
        for j in 0..<neurons.count {
            let output = outNeuron(neurons[j], input: input.body)
            neurons[j].output = function.activation(input: output)
            newInput.append(neurons[j].output)
        }
        return DataPiece(size: input.size, body: newInput)
    }
    
}

class Dropout: Layer {
    var probability: Float
    
    private enum CodingKeys: String, CodingKey {
        case probability
    }
    
    override func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(probability, forKey: .probability)
        try super.encode(to: encoder)
    }
    
    init(inputSize: Int, probability: Float) {
        self.probability = probability
        super.init(function: Plain())
        for _ in 0..<inputSize {
            neurons.append(Neuron(weights: [], bias: 0.0, delta: 0.0, output: 0.0))
        }
    }
    
    required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.probability = try container.decode(Float.self, forKey: .probability)
        try super.init(from: decoder)
    }
    
    override func forward(input: DataPiece, dropoutEnabled: Bool) -> DataPiece {
        var output = input
        for j in 0..<input.body.count {
            if dropoutEnabled {
                if Float.random(in: 0...1) < probability {
                    output.body[j] = 0
                }
            }
            neurons[j].output = input.body[j]
        }
        return output
    }
}

func getActivationFunction(rawValue: Int) -> ActivationFunction {
    switch rawValue {
    case ActivationFunctionRaw.reLU.rawValue:
        return ReLU()
    case ActivationFunctionRaw.sigmoid.rawValue:
        return Sigmoid()
    case ActivationFunctionRaw.plain.rawValue:
        return Plain()
    default:
        fatalError()
    }
}

enum ActivationFunctionRaw: Int {
    case sigmoid = 0
    case reLU
    case plain
}

protocol ActivationFunction: Codable {
    var rawValue: Int { get }
    func activation(input: Float) -> Float
}

struct Sigmoid: ActivationFunction, Codable {
    var rawValue: Int = 0
    func activation(input: Float) -> Float {
        return 1.0/(1.0+exp(-input))
    }
}

struct ReLU: ActivationFunction, Codable {
    var rawValue: Int = 1
    func activation(input: Float) -> Float {
        return max(Float.zero, input)
    }
}

struct Plain: ActivationFunction, Codable {
    var rawValue: Int = 2
    func activation(input: Float) -> Float {
        return input
    }
}
