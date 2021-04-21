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
    var output: DataPiece?
    
    private enum CodingKeys: String, CodingKey {
        case neurons
        case function
        case output
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(function.rawValue, forKey: .function)
        try container.encode(neurons, forKey: .neurons)
        try container.encode(output, forKey: .output)
    }
    
    public required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let activationRaw = try container.decode(Int.self, forKey: .function)
        function = getActivationFunction(rawValue: activationRaw)
        neurons = try container.decode([Neuron].self, forKey: .neurons)
        output = try container.decode(DataPiece.self, forKey: .output)
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
        output.withUnsafeMutableBufferPointer { outputPtr in
            kernel.withUnsafeBufferPointer { kernelPtr in
                DispatchQueue.concurrentPerform(iterations: outputPtr.count, execute: { i in
                    outputPtr[i].withUnsafeMutableBufferPointer { outputRowPtr in
                        kernelPtr[i].withUnsafeBufferPointer { kernelRowPtr in
                            DispatchQueue.concurrentPerform(iterations: outputPtr[i].count, execute: { j in
                                outputRowPtr[j] *= kernelRowPtr[j]
                            })
                        }
                    }
                })
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
    
    public init(inputSize: Int) {
        super.init(function: Plain())
        output = DataPiece(size: .init(width: inputSize), body: Array(repeating: Float.zero, count: inputSize))
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
        output?.body = input.body
        output?.size = .init(width: newWidth)
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
        var tempOutput = Array(repeating: Array(repeating: Array(repeating: Float.zero, count: outputSize.depth!), count: outputSize.width), count: outputSize.height!)
        var tempY = 0, outY = 0
        tempOutput.withUnsafeMutableBufferPointer { outputPtr in
            filters.withUnsafeBufferPointer { filtersPtr in
                while tempY + kernelSize <= inputSize.height! {
                    var tempX = 0, outX = 0
                    while tempX + kernelSize <= inputSize.width {
                        DispatchQueue.concurrentPerform(iterations: filtersPtr.count, execute: { i in
                            DispatchQueue.concurrentPerform(iterations: inputSize.depth!, execute: { j in
                                var piece = [[Float]]()
                                for y in tempY ..< tempY + kernelSize {
                                    var column = [Float]()
                                    for x in tempX ..< tempX + kernelSize {
                                        column.append(lastInput!.get(x: x, y: y, z: j))
                                    }
                                    piece.append(column)
                                }
                                outputPtr[outY][outX][i*inputSize.depth! + j] = function.activation(input: matrixSum(matrix: filtersPtr[i].apply(to: piece)))
                            })
                        })
                        tempX += stride
                        outX += 1
                    }
                    tempY += stride
                    outY += 1
                }
            }
        }
        var flat = [Float]()
        for i in tempOutput {
            for j in i {
                flat.append(contentsOf: j)
            }
        }
        self.output = DataPiece(size: outputSize, body: flat)
        return self.output!
    }
    
    override func backward(input: DataPiece, previous: Layer?) -> DataPiece {
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
        var output = Array(repeating: [[Float]](), count: filters.count)
        filters.withUnsafeBufferPointer { filtersPtr in
            output.withUnsafeMutableBufferPointer { outputPtr in
                filterErrors.withUnsafeMutableBufferPointer { filterErrorsPtr in
                    DispatchQueue.concurrentPerform(iterations: filters.count, execute: { filter in
                        filtersPtr[filter].kernel.withUnsafeBufferPointer { kernelPtr in
                            var error = Float.zero
                            var tempY = 0, outY = 0
                            var filterOutput = Array(repeating: Array(repeating: Float.zero, count: inputSize.height!), count: inputSize.width)
                            filterOutput.withUnsafeMutableBufferPointer { filterOutputPtr in
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
                                        var tempKernel = filtersPtr[filter].kernel
                                        for i in 0..<tempKernel.count {
                                            for j in 0..<tempKernel[i].count {
                                                tempKernel[i][j] *= resizedInput.get(x: outX, y: outY)
                                            }
                                        }
                                        for y in tempY ..< tempY + kernelSize {
                                            for x in tempX ..< tempX + kernelSize {
                                                filterOutputPtr[y][x] += resizedInput.get(x: outX, y: outY) * kernelPtr[y-tempY][x-tempX]
                                            }
                                        }
                                        tempX += stride
                                        outX += 1
                                    }
                                    tempY += stride
                                    outY += 1
                                }
                            }
                            filterErrorsPtr[filter] = error
                            outputPtr[filter] = filterOutput
                        }
                    })
                }
            }
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
    
    private let queue = DispatchQueue.global(qos: .userInitiated)
    
    public init(inputSize: Int, neuronsCount: Int, functionRaw: ActivationFunctionRaw) {
        let function = getActivationFunction(rawValue: functionRaw.rawValue)
        super.init(function: function)
        output = .init(size: .init(width: neuronsCount), body: Array(repeating: Float.zero, count: neuronsCount))
        
        for _ in 0..<neuronsCount {
            var weights = [Float]()
            for _ in 0..<inputSize {
                weights.append(Float.random(in: -1.0 ... 1.0))
            }
            neurons.append(Neuron(weights: weights, weightsDelta: .init(repeating: Float.zero, count: weights.count), bias: 0.0, delta: 0.0))
        }
    }
    
    private enum CodingKeys: String, CodingKey {
        case output
    }
    
    public override func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(output, forKey: .output)
        try super.encode(to: encoder)
    }
    
    public required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try super.init(from: decoder)
        output = try container.decode(DataPiece.self, forKey: .output)
    }
    
    override func forward(input: DataPiece, dropoutEnabled: Bool) -> DataPiece {
        output?.body.withUnsafeMutableBufferPointer { outputPtr in
            neurons.withUnsafeBufferPointer { neuronsPtr in
                DispatchQueue.concurrentPerform(iterations: neuronsPtr.count, execute: { i in
                    outputPtr[i] = function.activation(input: outNeuron(neuronsPtr[i], input: input.body))
                })
            }
        }
        return output!
    }
    
    override func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        var errors = Array(repeating: Float.zero, count: neurons.count)
        if let previous = previous {
            for j in 0..<neurons.count {
                for neuron in previous.neurons {
                    errors[j] += neuron.weights[j]*neuron.delta
                }
            }
        } else {
            for j in 0..<neurons.count {
                errors[j] = input.body[j] - output!.body[j]
            }
        }
        for j in 0..<neurons.count {
            neurons[j].delta = errors[j] * function.derivation(output: output!.body[j])
        }
        return output!
    }
    
    override func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
        let neuronsCount = neurons.count
        neurons.withUnsafeMutableBufferPointer { neuronsPtr in
            input.body.withUnsafeBufferPointer { inputPtr in
                DispatchQueue.concurrentPerform(iterations: neuronsCount, execute: { i in
                    neuronsPtr[i].weightsDelta.withUnsafeMutableBufferPointer { deltaPtr in
                        let weightsCount = deltaPtr.count
                        DispatchQueue.concurrentPerform(iterations: weightsCount, execute: { j in
                            deltaPtr[j] += learningRate * neuronsPtr[i].delta * inputPtr[j]
                        })
                        neuronsPtr[i].bias += learningRate * neuronsPtr[i].delta
                    }
                })
            }
        }
        return output!
    }
    
    override func updateWeights() {
        let neuronsCount = neurons.count
        neurons.withUnsafeMutableBufferPointer { neuronsPtr in
            DispatchQueue.concurrentPerform(iterations: neuronsCount, execute: { i in
                neuronsPtr[i].weights.withUnsafeMutableBufferPointer { weightsPtr in
                    neuronsPtr[i].weightsDelta.withUnsafeMutableBufferPointer { deltaPtr in
                        let weightsCount = deltaPtr.count
                        DispatchQueue.concurrentPerform(iterations: weightsCount, execute: { j in
                            weightsPtr[j] += deltaPtr[j]
                            deltaPtr[j] = 0
                        })
                    }
                }
            })
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
            neurons.append(Neuron(weights: [], weightsDelta: [], bias: 0.0, delta: 0.0))
        }
        output = DataPiece(size: .init(width: inputSize), body: Array(repeating: Float.zero, count: inputSize))
        #warning("Add 2D and 3D support")
    }
    
    public required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.probability = try container.decode(Float.self, forKey: .probability)
        self.cache = try container.decode([Bool].self, forKey: .cache)
        try super.init(from: decoder)
    }
    
    override func forward(input: DataPiece, dropoutEnabled: Bool) -> DataPiece {
        output = input
        for j in 0..<output!.body.count {
            if dropoutEnabled {
                if Float.random(in: 0...1) < probability {
                    cache[j] = false
                    output!.body[j] = 0
                } else {
                    cache[j] = true
                }
            }
        }
        return output!
    }
    
    override func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        for i in 0..<neurons.count {
            if !cache[i] {
                output?.body[i] = 0
            }
        }
        #warning("Check this")
        return output!
    }
    
    override func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
        return output!
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
