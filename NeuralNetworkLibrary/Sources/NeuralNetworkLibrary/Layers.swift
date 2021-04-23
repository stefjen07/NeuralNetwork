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
        case pooling2d
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
        case let payload as Pooling2D:
            try container.encode(Base.pooling2d, forKey: .base)
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
        case .pooling2d:
            self.layer = try container.decode(Pooling2D.self, forKey: .payload)
        }
    }

}

public class Layer: Codable {
    var neurons: [Neuron] = []
    fileprivate var function: ActivationFunction
    fileprivate var output: DataPiece?
    
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
    
    fileprivate init(function: ActivationFunction) {
        self.function = function
    }
}

class Filter: Codable {
    var kernel: [Float]
    var delta: [Float]
    
    public init(kernelSize: Int) {
        kernel = []
        delta = Array(repeating: Float.zero, count: kernelSize*kernelSize)
        for _ in 0..<kernelSize*kernelSize {
            kernel.append(Float.random(in: -1...1))
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
        output = DataPiece(size: outputSize, body: Array(repeating: Float.zero, count: outputSize.width*outputSize.height!*outputSize.depth!))
        output!.body.withUnsafeMutableBufferPointer { outputPtr in
            filters.withUnsafeBufferPointer { filtersPtr in
                DispatchQueue.concurrentPerform(iterations: outputSize.height!, execute: { outY in
                    let tempY = outY * stride
                    DispatchQueue.concurrentPerform(iterations: outputSize.width, execute: { outX in
                        let tempX = outX * stride
                        DispatchQueue.concurrentPerform(iterations: filtersPtr.count, execute: { i in
                            filtersPtr[i].kernel.withUnsafeBufferPointer { kernelPtr in
                                DispatchQueue.concurrentPerform(iterations: inputSize.depth!, execute: { j in
                                    var piece = Float.zero
                                    for y in tempY ..< tempY + kernelSize {
                                        for x in tempX ..< tempX + kernelSize {
                                            piece += kernelPtr[(y-tempY)*kernelSize+x-tempX] * lastInput!.get(x: x, y: y, z: j)
                                        }
                                    }
                                    outputPtr[outY*outputSize.width*outputSize.depth!+outX*outputSize.depth!+i*inputSize.depth!+j] = function.activation(input: piece)
                                })
                            }
                        })
                    })
                })
            }
        }
        return output!
    }
    
    override func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        guard let lastInput = lastInput else {
            fatalError("Backward propagation executed before forward propagation.")
        }
        let inputSize = lastInput.size
        let outputSize = DataSize(width: (inputSize.width - kernelSize) / stride + 1, height: (inputSize.height! - kernelSize) / stride + 1, depth: filters.count*inputSize.depth!)
        var resizedInput = input
        for i in 0..<resizedInput.body.count {
            resizedInput.body[i] = function.derivative(output: resizedInput.body[i])
        }
        resizedInput.size = outputSize
        var output = Array(repeating: Float.zero, count: outputSize.width * outputSize.height! * outputSize.depth!)
        filters.withUnsafeBufferPointer { filtersPtr in
            output.withUnsafeMutableBufferPointer { outputPtr in
                filterErrors.withUnsafeMutableBufferPointer { filterErrorsPtr in
                    DispatchQueue.concurrentPerform(iterations: filtersPtr.count, execute: { filter in
                        filtersPtr[filter].kernel.withUnsafeBufferPointer { kernelPtr in
                            var error = Float.zero
                            DispatchQueue.concurrentPerform(iterations: outputSize.height!, execute: { outY in
                                let tempY = outY * stride
                                DispatchQueue.concurrentPerform(iterations: outputSize.width, execute: { outX in
                                    let tempX = outX * stride
                                    DispatchQueue.concurrentPerform(iterations: inputSize.depth!, execute: { j in
                                        var piece = Float.zero
                                        for y in tempY ..< tempY + kernelSize {
                                            for x in tempX ..< tempX + kernelSize {
                                                piece += lastInput.get(x: x, y: y, z: j)
                                            }
                                        }
                                        error += piece * resizedInput.get(x: outX, y: outY)
                                        var tempKernel = filtersPtr[filter].kernel
                                        for i in 0..<tempKernel.count {
                                            tempKernel[i] *= resizedInput.get(x: outX, y: outY)
                                        }
                                        let cof = resizedInput.get(x: outX, y: outY)
                                        for y in tempY ..< tempY + kernelSize {
                                            for x in tempX ..< tempX + kernelSize {
                                                outputPtr[outY*outputSize.width*outputSize.depth!+outX*outputSize.depth!+filter*inputSize.depth!+j] += cof * kernelPtr[(y-tempY)*kernelSize+x-tempX]
                                            }
                                        }
                                    })
                                })
                            })
                            filterErrorsPtr[filter] = error
                        }
                    })
                }
            }
        }
        return DataPiece(size: .init(width: inputSize.height!, height: inputSize.width, depth: filters.count*inputSize.depth!), body: output)
    }
    
    override func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
        for i in 0..<filters.count {
            for j in 0..<filters[i].kernel.count {
                filters[i].delta[j] += learningRate * filterErrors[i]
            }
        }
        return output!
    }
    
    override func updateWeights() {
        for i in 0..<filters.count {
            for j in 0..<filters[i].kernel.count {
                filters[i].kernel[j] += filters[i].delta[j]
            }
        }
    }
    
}

public enum PoolingMode: Int {
    case average = 0
    case max
}

public class Pooling2D: Layer {
    var kernelSize: Int
    var stride: Int
    var mode: PoolingMode
    private var lastInput: DataPiece?
    
    private enum CodingKeys: String, CodingKey {
        case kernelSize
        case stride
        case mode
    }
    
    public init(kernelSize: Int, stride: Int, mode: PoolingMode, functionRaw: ActivationFunctionRaw) {
        let function = getActivationFunction(rawValue: functionRaw.rawValue)
        self.kernelSize = kernelSize
        self.stride = stride
        self.mode = mode
        super.init(function: function)
    }
    
    required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        kernelSize = try container.decode(Int.self, forKey: .kernelSize)
        stride = try container.decode(Int.self, forKey: .stride)
        mode = PoolingMode(rawValue: try container.decode(Int.self, forKey: .mode)) ?? .average
        try super.init(from: decoder)
    }
    
    public override func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(kernelSize, forKey: .kernelSize)
        try container.encode(stride, forKey: .stride)
        try container.encode(mode.rawValue, forKey: .mode)
        try super.encode(to: encoder)
    }
    
    override func forward(input: DataPiece, dropoutEnabled: Bool) -> DataPiece {
        lastInput = input
        if input.size.type == .twoD {
            lastInput?.size = DataSize(width: input.size.width, height: input.size.height!, depth: 1)
        } else if input.size.type == .oneD {
            fatalError("Pooling 2D input must be at least two-dimensional.")
        }
        
        let inputSize = lastInput!.size
        let outputSize = DataSize(width: (inputSize.width - kernelSize) / stride + 1, height: (inputSize.height! - kernelSize) / stride + 1, depth: inputSize.depth!)
        output = DataPiece(size: outputSize, body: Array(repeating: Float.zero, count: outputSize.width*outputSize.height!*outputSize.depth!))
        output!.body.withUnsafeMutableBufferPointer { outputPtr in
            DispatchQueue.concurrentPerform(iterations: outputSize.height!, execute: { outY in
                let tempY = outY * stride
                DispatchQueue.concurrentPerform(iterations: outputSize.width, execute: { outX in
                    let tempX = outX * stride
                    DispatchQueue.concurrentPerform(iterations: inputSize.depth!, execute: { i in
                        var piece = mode == .max ? lastInput!.get(x: tempX, y: tempY, z: i) : Float.zero
                        for y in tempY ..< tempY + kernelSize {
                            for x in tempX ..< tempX + kernelSize {
                                if mode == .max {
                                    piece = max(piece, lastInput!.get(x: x, y: y, z: i))
                                } else {
                                    piece += lastInput!.get(x: x, y: y, z: i)
                                }
                            }
                        }
                        if mode == .average {
                            piece /= Float(kernelSize * kernelSize)
                        }
                        outputPtr[outY*outputSize.width*outputSize.depth!+outX*outputSize.depth!+i] = function.activation(input: piece)
                    })
                })
            })
        }
        return output!
    }
    
    override func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        guard let lastInput = lastInput else {
            fatalError("Backward propagation executed before forward propagation.")
        }
        let inputSize = lastInput.size
        let outputSize = DataSize(width: (inputSize.width - kernelSize) / stride + 1, height: (inputSize.height! - kernelSize) / stride + 1, depth: inputSize.depth!)
        var resizedInput = input
        for i in 0..<resizedInput.body.count {
            resizedInput.body[i] = function.derivative(output: resizedInput.body[i])
        }
        resizedInput.size = outputSize
        output = DataPiece(size: inputSize, body: Array(repeating: Float.zero, count: inputSize.width*inputSize.height!*inputSize.depth!))
        output!.body.withUnsafeMutableBufferPointer { outputPtr in
            DispatchQueue.concurrentPerform(iterations: outputSize.height!, execute: { outY in
                let tempY = outY * stride
                DispatchQueue.concurrentPerform(iterations: outputSize.width, execute: { outX in
                    let tempX = outX * stride
                    DispatchQueue.concurrentPerform(iterations: outputSize.depth!, execute: { j in
                        let val = resizedInput.get(x: outX, y: outY, z: j) / Float(kernelSize * kernelSize)
                        if mode == .average {
                            for y in tempY ..< tempY + kernelSize {
                                for x in tempX ..< tempX + kernelSize {
                                    outputPtr[y*inputSize.width*inputSize.depth!+x*inputSize.depth!+j] += val
                                }
                            }
                        } else {
                            var maxV = lastInput.get(x: tempX, y: tempY, z: j)
                            for y in tempY ..< tempY + kernelSize {
                                for x in tempX ..< tempX + kernelSize {
                                    maxV = max(maxV, lastInput.get(x: x, y: y, z: j))
                                }
                            }
                            for y in tempY ..< tempY + kernelSize {
                                for x in tempX ..< tempX + kernelSize {
                                    if lastInput.get(x: x, y: y, z: j) == maxV {
                                        outputPtr[y*inputSize.width*inputSize.depth!+x*inputSize.depth!+j] += val
                                    }
                                }
                            }
                        }
                    })
                })
            })
        }
        return output!
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
    
    public required init(from decoder: Decoder) throws {
        try super.init(from: decoder)
    }
    
    override func forward(input: DataPiece, dropoutEnabled: Bool) -> DataPiece {
        input.body.withUnsafeBufferPointer { inputPtr in
            output?.body.withUnsafeMutableBufferPointer { outputPtr in
                neurons.withUnsafeBufferPointer { neuronsPtr in
                    DispatchQueue.concurrentPerform(iterations: neuronsPtr.count, execute: { i in
                        var out = neuronsPtr[i].bias
                        neuronsPtr[i].weights.withUnsafeBufferPointer { weightsPtr in
                            DispatchQueue.concurrentPerform(iterations: neuronsPtr[i].weights.count, execute: { i in
                                out += weightsPtr[i] * inputPtr[i]
                            })
                        }
                        outputPtr[i] = function.activation(input: out)
                    })
                }
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
            neurons[j].delta = errors[j] * function.derivative(output: output!.body[j])
        }
        return output!
    }
    
    override func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
        neurons.withUnsafeMutableBufferPointer { neuronsPtr in
            input.body.withUnsafeBufferPointer { inputPtr in
                DispatchQueue.concurrentPerform(iterations: neuronsPtr.count, execute: { i in
                    neuronsPtr[i].weightsDelta.withUnsafeMutableBufferPointer { deltaPtr in
                        DispatchQueue.concurrentPerform(iterations: deltaPtr.count, execute: { j in
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
    var probability: Int
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
    
    public init(inputSize: Int, probability: Int) {
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
        self.probability = try container.decode(Int.self, forKey: .probability)
        self.cache = try container.decode([Bool].self, forKey: .cache)
        try super.init(from: decoder)
    }
    
    override func forward(input: DataPiece, dropoutEnabled: Bool) -> DataPiece {
        output = input
        if dropoutEnabled {
            cache.withUnsafeMutableBufferPointer { cachePtr in
                output?.body.withUnsafeMutableBufferPointer { outputPtr in
                    DispatchQueue.concurrentPerform(iterations: outputPtr.count, execute: { i in
                        if Int.random(in: 0...100) < probability {
                            cachePtr[i] = false
                            outputPtr[i] = 0
                        } else {
                            cachePtr[i] = true
                        }
                    })
                }
            }
        }
        return output!
    }
    
    override func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        return output!
    }
    
    override func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
        return output!
    }
}

fileprivate func getActivationFunction(rawValue: Int) -> ActivationFunction {
    switch rawValue {
    case ActivationFunctionRaw.reLU.rawValue:
        return ReLU()
    case ActivationFunctionRaw.sigmoid.rawValue:
        return Sigmoid()
    default:
        return Plain()
    }
}

public enum ActivationFunctionRaw: Int {
    case sigmoid = 0
    case reLU
    case plain
}

protocol ActivationFunction: Codable {
    var rawValue: Int { get }
    func activation(input: Float) -> Float
    func derivative(output: Float) -> Float
}

fileprivate struct Sigmoid: ActivationFunction, Codable {
    public var rawValue: Int = 0
    
    public func activation(input: Float) -> Float {
        return 1.0/(1.0+exp(-input))
    }
    
    public func derivative(output: Float) -> Float {
        return output * (1.0-output)
    }
}

fileprivate struct ReLU: ActivationFunction, Codable {
    public var rawValue: Int = 1
    
    public func activation(input: Float) -> Float {
        return max(Float.zero, input)
    }
    
    public func derivative(output: Float) -> Float {
        return output < 0 ? 0 : 1
    }
}

fileprivate struct Plain: ActivationFunction, Codable {
    public var rawValue: Int = 2
    
    public func activation(input: Float) -> Float {
        return input
    }
    
    public func derivative(output: Float) -> Float {
        return 1
    }
}

#if DEBUG
func getActivationFunctionMirror(rawValue: Int) -> ActivationFunction {
    getActivationFunction(rawValue: rawValue)
}
#endif
