//
//  Layers.swift
//  NeuralNetwork
//
//  Created by Евгений on 15.04.2021.
//

import Foundation

public struct LayerWrapper: Codable {
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
    
    public func encode(to encoder: Encoder) throws {
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
    
    public init(from decoder: Decoder) throws {
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

public class Layer: Codable {
    var neurons: [Neuron] = []
    var function: ActivationFunction
    
    private enum CodingKeys: String, CodingKey {
        case neurons
        case function
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(function.rawValue, forKey: .function)
        try container.encode(neurons, forKey: .neurons)
    }
    
    public required init(from decoder: Decoder) throws {
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
    
    func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
        return input
    }
    
    func updateWeights() {
        return
    }
    
    public init(function: ActivationFunction) {
        self.function = function
    }
}

class Filter: Codable {
    var kernel: [[Float]]
    var output: [[Float]]
    var delta: [[Float]]
    
    func apply(to piece: [[Float]]) -> [[Float]] {
        if piece.count != kernel.count {
            fatalError("Piece size must be equal kernel size.")
        }
        output = piece
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
    
    public init(kernelSize: Int) {
        kernel = []
        output = []
        delta = []
        for _ in 0..<kernelSize {
            var row = [Float](), deltaRow = [Float]()
            for _ in 0..<kernelSize {
                row.append(Float.random(in: -1...1))
                deltaRow.append(Float.zero)
            }
            kernel.append(row)
            delta.append(deltaRow)
        }
    }
}

public class Flatten: Layer {
    private var output: DataPiece?
    
    public init() {
        super.init(function: Plain())
    }
    
    public required init(from decoder: Decoder) throws {
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
        output = input
        output!.size = .init(width: newWidth)
        return output!
    }
    
    override func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        return output!
    }
    
    override func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
        return output!
    }
}

public class Convolutional2D: Layer {
    var filters: [Filter]
    var filterErrors: [Float]
    var kernelSize: Int
    var stride: Int
    private var output: DataPiece?
    private var lastInput: DataPiece?
    
    private enum CodingKeys: String, CodingKey {
        case filters
        case kernelSize
        case stride
        case errors
    }
    
    public override func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(filters, forKey: .filters)
        try container.encode(kernelSize, forKey: .kernelSize)
        try container.encode(stride, forKey: .stride)
        try container.encode(filterErrors, forKey: .errors)
        try super.encode(to: encoder)
    }
    
    public required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.filters = try container.decode([Filter].self, forKey: .filters)
        self.kernelSize = try container.decode(Int.self, forKey: .kernelSize)
        self.stride = try container.decode(Int.self, forKey: .stride)
        self.filterErrors = try container.decode([Float].self, forKey: .errors)
        try super.init(from: decoder)
    }
    
    public init(filters: Int, kernelSize: Int, stride: Int, functionRaw: ActivationFunctionRaw) {
        let function = getActivationFunction(rawValue: functionRaw.rawValue)
        self.filters = []
        self.kernelSize = kernelSize
        self.stride = stride
        for _ in 0..<filters {
            self.filters.append(Filter(kernelSize: kernelSize))
        }
        self.filterErrors = Array(repeating: Float.zero, count: filters)
        super.init(function: function)
    }
    
    override func forward(input: DataPiece, dropoutEnabled: Bool) -> DataPiece {
        lastInput = input
        if input.size.type == .twoD {
            lastInput?.size = DataSize(width: input.size.width, height: input.size.height!, depth: 1)
        } else if input.size.type == .oneD {
            fatalError("Convolutional 2D input must be at least two-dimensional.")
        }
        
        let inputSize = lastInput!.size
        let outputSize = DataSize(width: (inputSize.width - kernelSize) / stride + 1, height: (inputSize.height! - kernelSize) / stride + 1, depth: filters.count*inputSize.depth!)
        var output = Array(repeating: Array(repeating: Array(repeating: Float.zero, count: outputSize.depth!), count: outputSize.width), count: outputSize.height!)
        var tempY = 0, outY = 0
        while tempY + kernelSize <= inputSize.height! {
            var tempX = 0, outX = 0
            while tempX + kernelSize <= inputSize.width {
                for filter in 0..<filters.count {
                    for depth in 0..<inputSize.depth! {
                        var piece = [[Float]]()
                        for y in tempY ..< tempY + kernelSize {
                            var column = [Float]()
                            for x in tempX ..< tempX + kernelSize {
                                column.append(lastInput!.get(x: x, y: y, z: depth))
                            }
                            piece.append(column)
                        }
                        output[outY][outX][filter*inputSize.depth! + depth] = function.activation(input: matrixSum(matrix: filters[filter].apply(to: piece)))
                    }
                }
                tempX += stride
                outX += 1
            }
            tempY += stride
            outY += 1
        }
        var flat = [Float]()
        for i in output {
            for j in i {
                flat.append(contentsOf: j)
            }
        }
        self.output = DataPiece(size: outputSize, body: flat)
        return self.output!
    }
    
    override func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        filterErrors.removeAll()
        guard let lastInput = lastInput else {
            fatalError("Backward propagation executed before forward propagation.")
        }
        let inputSize = lastInput.size
        let outputSize = DataSize(width: (inputSize.width - kernelSize) / stride + 1, height: (inputSize.height! - kernelSize) / stride + 1)
        var resizedInput = input
        for i in 0..<resizedInput.body.count {
            resizedInput.body[i] = function.derivation(output: resizedInput.body[i])
        }
        resizedInput.size = outputSize
        var output = [[[Float]]]()
        for filter in filters {
            var error = Float.zero
            var tempY = 0, outY = 0
            var filterOutput = Array(repeating: Array(repeating: Float.zero, count: inputSize.height!), count: inputSize.width)
            while tempY + kernelSize <= inputSize.height! {
                var tempX = 0, outX = 0
                while tempX + kernelSize <= inputSize.width {
                    var piece = [[Float]]()
                    for y in tempY ..< tempY + kernelSize {
                        var column = [Float]()
                        for x in tempX ..< tempX + kernelSize {
                            column.append(lastInput.get(x: x, y: y))
                        }
                        piece.append(column)
                    }
                    error += matrixSum(matrix: piece) * resizedInput.get(x: outX, y: outY)
                    var tempKernel = filter.kernel
                    for i in 0..<tempKernel.count {
                        for j in 0..<tempKernel[i].count {
                            tempKernel[i][j] *= resizedInput.get(x: outX, y: outY)
                        }
                    }
                    for y in tempY ..< tempY + kernelSize {
                        for x in tempX ..< tempX + kernelSize {
                            filterOutput[y][x] += resizedInput.get(x: outX, y: outY) * filter.kernel[y-tempY][x-tempX]
                        }
                    }
                    tempX += stride
                    outX += 1
                }
                tempY += stride
                outY += 1
            }
            filterErrors.append(error)
            output.append(filterOutput)
        }
        return DataPiece(size: .init(width: output[0].count, height: output[0][0].count, depth: filters.count), body: output.flatMap { $0.flatMap { $0 } })
    }
    
    override func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
        for i in 0..<filters.count {
            for x in 0..<filters[i].kernel.count {
                for y in 0..<filters[i].kernel[x].count {
                    filters[i].delta[x][y] += learningRate * filterErrors[i]
                }
            }
        }
        return output!
    }
    
    override func updateWeights() {
        for i in 0..<filters.count {
            for x in 0..<filters[i].kernel.count {
                for y in 0..<filters[i].kernel[x].count {
                    filters[i].kernel[x][y] += filters[i].delta[x][y]
                }
            }
        }
    }
    
}

public class Dense: Layer {
    
    public init(inputSize: Int, neuronsCount: Int, functionRaw: ActivationFunctionRaw) {
        let function = getActivationFunction(rawValue: functionRaw.rawValue)
        
        super.init(function: function)
        
        for _ in 0..<neuronsCount {
            var weights = [Float]()
            for _ in 0..<inputSize {
                weights.append(Float.random(in: -1.0 ... 1.0))
            }
            neurons.append(Neuron(weights: weights, weightsDelta: .init(repeating: Float.zero, count: weights.count), bias: 0.0, delta: 0.0, output: 0.0))
        }
    }
    
    public required init(from decoder: Decoder) throws {
        try super.init(from: decoder)
    }
    
    override func forward(input: DataPiece, dropoutEnabled: Bool) -> DataPiece {
        var newInput = [Float]()
        for j in 0..<neurons.count {
            let output = outNeuron(neurons[j], input: input.body)
            neurons[j].output = function.activation(input: output)
            newInput.append(neurons[j].output)
        }
        return DataPiece(size: .init(width: newInput.count), body: newInput)
    }
    
    override func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        var errors = [Float]()
        if let previous = previous {
            for j in 0..<neurons.count {
                var error = Float.zero
                for neuron in previous.neurons {
                    error += neuron.weights[j]*neuron.delta
                }
                errors.append(error)
            }
        } else {
            for j in 0..<neurons.count {
                errors.append(input.body[j]-neurons[j].output)
            }
        }
        for j in 0..<neurons.count {
            neurons[j].delta = errors[j] * function.derivation(output: neurons[j].output)
        }
        return DataPiece(size: .init(width: neurons.count), body: neurons.map { $0.output })
    }
    
    override func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
        for j in 0..<neurons.count {
            for m in 0..<neurons[j].weights.count {
                neurons[j].weightsDelta[m] += learningRate * neurons[j].delta * input.body[m]
            }
            neurons[j].bias += learningRate * neurons[j].delta
        }
        return DataPiece(size: .init(width: neurons.count), body: neurons.map { $0.output })
    }
    
    override func updateWeights() {
        for i in 0..<neurons.count {
            for j in 0..<neurons[i].weightsDelta.count {
                neurons[i].weights[j] += neurons[i].weightsDelta[j]
                neurons[i].weightsDelta[j] = 0
            }
        }
    }
    
}

public class Dropout: Layer {
    var probability: Float
    var cache: [Bool]
    
    private enum CodingKeys: String, CodingKey {
        case probability
        case cache
    }
    
    public override func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(probability, forKey: .probability)
        try container.encode(cache, forKey: .cache)
        try super.encode(to: encoder)
    }
    
    public init(inputSize: Int, probability: Float) {
        self.probability = probability
        self.cache = Array(repeating: true, count: inputSize)
        super.init(function: Plain())
        for _ in 0..<inputSize {
            neurons.append(Neuron(weights: [], weightsDelta: [], bias: 0.0, delta: 0.0, output: 0.0))
        }
    }
    
    public required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.probability = try container.decode(Float.self, forKey: .probability)
        self.cache = try container.decode([Bool].self, forKey: .cache)
        try super.init(from: decoder)
    }
    
    override func forward(input: DataPiece, dropoutEnabled: Bool) -> DataPiece {
        var output = input
        for j in 0..<output.body.count {
            if dropoutEnabled {
                if Float.random(in: 0...1) < probability {
                    cache[j] = false
                    output.body[j] = 0
                } else {
                    cache[j] = true
                }
            }
            neurons[j].output = output.body[j]
        }
        return output
    }
    
    override func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        var output = DataPiece(size: .init(width: neurons.count), body: neurons.map { $0.output })
        for i in 0..<neurons.count {
            if !cache[i] {
                output.body[i] = 0
            }
        }
        return output
    }
    
    override func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
        return DataPiece(size: .init(width: neurons.count), body: neurons.map { $0.output })
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

public enum ActivationFunctionRaw: Int {
    case sigmoid = 0
    case reLU
    case plain
}

public protocol ActivationFunction: Codable {
    var rawValue: Int { get }
    func activation(input: Float) -> Float
    func derivation(output: Float) -> Float
}

public struct Sigmoid: ActivationFunction, Codable {
    public var rawValue: Int = 0
    
    public func activation(input: Float) -> Float {
        return 1.0/(1.0+exp(-input))
    }
    
    public func derivation(output: Float) -> Float {
        return output * (1.0-output)
    }
}

public struct ReLU: ActivationFunction, Codable {
    public var rawValue: Int = 1
    
    public func activation(input: Float) -> Float {
        return max(Float.zero, input)
    }
    
    public func derivation(output: Float) -> Float {
        return output < 0 ? 0 : 1
    }
}

public struct Plain: ActivationFunction, Codable {
    public var rawValue: Int = 2
    
    public func activation(input: Float) -> Float {
        return input
    }
    
    public func derivation(output: Float) -> Float {
        return 1
    }
}
