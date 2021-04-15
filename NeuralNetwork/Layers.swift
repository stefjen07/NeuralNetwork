//
//  Layers.swift
//  NeuralNetwork
//
//  Created by Евгений on 15.04.2021.
//

import Foundation

protocol Layer: Codable {
    var neurons: [Neuron] { get set }
    var function: ActivationFunction { get set }
}

struct LayerWrapper: Codable {
    let layer: Layer
    
    private enum CodingKeys: String, CodingKey {
        case base
        case payload
    }

    private enum Base: Int, Codable {
        case dense = 0
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
        }
    }

}

struct Dense: Layer {
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
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let activationRaw = try container.decode(Int.self, forKey: .function)
        function = getActivationFunction(rawValue: activationRaw)
        neurons = try container.decode([Neuron].self, forKey: CodingKeys.neurons)
    }
    
    init(neuronsCount: Int, inputSize: Int, function: ActivationFunctionRaw) {
        for _ in 0..<neuronsCount {
            var weights = [Float]()
            for _ in 0..<inputSize {
                weights.append(Float.random(in: -1.0 ... 1.0))
            }
            neurons.append(Neuron(weights: weights, bias: 0.0, delta: 0.0, output: 0.0))
        }
        self.function = getActivationFunction(rawValue: function.rawValue)
    }
}

func getActivationFunction(rawValue: Int) -> ActivationFunction {
    switch rawValue {
    case ActivationFunctionRaw.reLU.rawValue:
        return ReLU()
    case ActivationFunctionRaw.sigmoid.rawValue:
        return Sigmoid()
    default:
        fatalError()
    }
}

enum ActivationFunctionRaw: Int {
    case sigmoid = 0
    case reLU
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

struct ReLU: ActivationFunction {
    var rawValue: Int = 1
    
    func activation(input: Float) -> Float {
        return max(Float.zero, input)
    }
}
