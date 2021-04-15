//
//  Core.swift
//  NeuralNetwork
//
//  Created by Евгений on 15.04.2021.
//

import Foundation

func transferDerivative(output: Float) -> Float {
    return output * (1.0-output)
}

func outNeuron(_ neuron: Neuron, input: [Float]) -> Float {
    var out = neuron.bias
    for i in 0..<neuron.weights.count {
        out += neuron.weights[i] * input[i]
    }
    return out
}

struct DataItem {
    var input: [Float]
    var output: [Float]
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
                for i in 0..<item.output.count {
                    error+=pow(item.output[i]-predictions[i], 2)
                }
                backward(expected: item.output)
                updateWeights(row: item.input)
            }
            print("Epoch \(epoch+1), error \(error).")
        }
    }
    
    func predict(input: [Float]) -> Int {
        dropoutEnabled = false
        let output = forward(networkInput: input)
        var maxi = 0
        for i in 1..<output.count {
            if(output[i]>output[maxi]) {
                maxi = i
            }
        }
        return maxi
    }
    
    func updateWeights(row: [Float]) {
        for i in 0..<layers.count {
            var input = row
            if i != 0 {
                input.removeAll()
                for neuron in layers[i-1].neurons {
                    input.append(neuron.output)
                }
            }
            for j in 0..<layers[i].neurons.count {
                for m in 0..<layers[i].neurons[j].weights.count {
                    layers[i].neurons[j].weights[m] += learningRate * layers[i].neurons[j].delta * input[m]
                }
                layers[i].neurons[j].bias += learningRate * layers[i].neurons[j].delta
            }
        }
    }
    
    func forward(networkInput: [Float]) -> [Float] {
        var input = networkInput, newInput = [Float]()
        for i in 0..<layers.count {
            newInput.removeAll()
            switch layers[i] {
            case let layer as Dropout:
                for j in 0..<input.count {
                    if dropoutEnabled {
                        if Float.random(in: 0...1) < layer.probability {
                            input[j] = 0
                        }
                    }
                    layers[i].neurons[j].output = input[j]
                }
            default:
                for j in 0..<layers[i].neurons.count {
                    let output = outNeuron(layers[i].neurons[j], input: input)
                    layers[i].neurons[j].output = layers[i].function.activation(input: output)
                    newInput.append(layers[i].neurons[j].output)
                }
                input = newInput
            }
        }
        return input
    }
    
    func backward(expected: [Float]) {
        for i in (0..<layers.count).reversed() {
            let layer = layers[i]
            if layer is Dropout {
                continue
            }
            var errors = [Float]()
            if i == layers.count-1 {
                for j in 0..<layer.neurons.count {
                    errors.append(expected[j]-layer.neurons[j].output)
                }
            } else {
                for j in 0..<layer.neurons.count {
                    var error = Float.zero
                    if !(layers[i+1] is Dropout) {
                        for neuron in layers[i+1].neurons {
                            error += neuron.weights[j]*neuron.delta
                        }
                    }
                    errors.append(error)
                }
            }
            for j in 0..<layer.neurons.count {
                layers[i].neurons[j].delta = errors[j] * transferDerivative(output: layers[i].neurons[j].output)
            }
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
