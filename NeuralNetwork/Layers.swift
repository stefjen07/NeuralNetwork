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
    
    init(function: ActivationFunction) {
        self.function = function
    }
}

/*class Conv2D: Layer {
    var size: CGSize
    
    private enum CodingKeys: String, CodingKey {
        case size
    }
    
    override func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(size, forKey: .size)
        try super.encode(to: encoder)
    }
    
    required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.size = try container.decode(CGSize.self, forKey: .size)
        
        try super.init(from: decoder)
    }
    
    init(inputSize: Int, size: CGSize, functionRaw: ActivationFunctionRaw) {
        let function = getActivationFunction(rawValue: functionRaw.rawValue)
        let neuronsCount = Int(size.width * size.height)
        self.size = size
        
        super.init(function: function)
        
        for _ in 0..<neuronsCount {
            var weights = [Float]()
            for _ in 0..<inputSize {
                weights.append(Float.random(in: -1.0 ... 1.0))
            }
            self.neurons.append(Neuron(weights: weights, bias: 0.0, delta: 0.0, output: 0.0))
        }
    }
    
}*/

class Dense: Layer {
    
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
