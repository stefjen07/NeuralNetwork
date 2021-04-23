    import XCTest
    @testable import NeuralNetworkLibrary

    final class NeuralNetworkLibraryTests: XCTestCase {
        
        func testClassifierOutput() throws {
            let classes = 5, correct = 2
            let expected: [Float] = [0, 0, 1, 0, 0]
            let result = classifierOutput(classes: classes, correct: correct)
            XCTAssert( result.body == expected && result.size.width == classes )
        }
        
        func testGetDataPieceCell() throws {
            var input = DataPiece(size: .init(width: 8), body: [0,1,2,3,4,5,6,7])
            XCTAssertEqual(input.get(x: 3), 3)
            input.size = .init(width: 4, height: 2)
            XCTAssertEqual(input.get(x: 1, y: 1), 5)
            input.size = .init(width: 2, height: 2, depth: 2)
            XCTAssertEqual(input.get(x: 1, y: 1, z: 0), 6)
        }
        
        func testDropoutLeak() throws {
            let dropout = Dropout(inputSize: 2, probability: 0)
            let dense = Dense(inputSize: 2, neuronsCount: 2, functionRaw: .sigmoid)
            let endDense = Dense(inputSize: 2, neuronsCount: 2, functionRaw: .sigmoid)
            let item = DataItem(input: DataPiece(size: .init(width: 2), body: [1, 1]), output: DataPiece(size: .init(width: 2), body: [0, 1]))
            
            let forwardDrop = endDense.forward(input: dropout.forward(input: dense.forward(input: item.output, dropoutEnabled: true), dropoutEnabled: true), dropoutEnabled: true)
            let forward = endDense.forward(input: dense.forward(input: item.output, dropoutEnabled: true), dropoutEnabled: true)
            XCTAssertEqual(forwardDrop, forward)
            
            let resDrop = dense.backward(input: dropout.backward(input: endDense.backward(input: item.input, previous: nil), previous: endDense), previous: endDense)
            let res = dense.backward(input: endDense.backward(input: item.input, previous: nil), previous: endDense)
            XCTAssertEqual(resDrop, res)
            
            let tempDense = dense
            let tempEnd = endDense
            let learningRate = Float(0.1)
            
            let _ = tempEnd.deltaWeights(input: dropout.deltaWeights(input: tempDense.deltaWeights(input: item.input, learningRate: learningRate), learningRate: learningRate), learningRate: learningRate)
            let _ = endDense.deltaWeights(input: dense.deltaWeights(input: item.input, learningRate: learningRate), learningRate: learningRate)
            
            tempEnd.updateWeights()
            endDense.updateWeights()
            
            XCTAssertEqual(tempEnd.neurons.map { $0.weights }, endDense.neurons.map { $0.weights })
        }
        
        func Plain() -> ActivationFunction {
            return getActivationFunctionMirror(rawValue: ActivationFunctionRaw.plain.rawValue)
        }
        
        func ReLU() -> ActivationFunction {
            return getActivationFunctionMirror(rawValue: ActivationFunctionRaw.reLU.rawValue)
        }
        
        func Sigmoid() -> ActivationFunction {
            return getActivationFunctionMirror(rawValue: ActivationFunctionRaw.sigmoid.rawValue)
        }
        
        func testFunctions() throws {
            let val = Float.random(in: -1000...1000)
            XCTAssertEqual(val / Plain().activation(input: val), Plain().derivative(output: val))
            XCTAssertEqual(val < 0 ? 0 : val, ReLU().activation(input: val))
            XCTAssertEqual(val < 0, ReLU().derivative(output: val) == 0)
            XCTAssertEqual(val >= 0, ReLU().derivative(output: 0-val) == 0)
            XCTAssertEqual(1.0/(1+exp(0-val)), Sigmoid().activation(input: val))
            XCTAssertEqual(val*(1-val), Sigmoid().derivative(output: val))
        }
        
        func testTrain() throws {
            let network = NeuralNetwork()
            network.learningRate = 0.5
            network.epochs = 1000
            network.layers = [
                Convolutional2D(filters: 1, kernelSize: 1, stride: 1, functionRaw: .sigmoid),
                Flatten(inputSize: 4),
                Dense(inputSize: 4, neuronsCount: 6, functionRaw: .sigmoid),
                Dense(inputSize: 6, neuronsCount: 2, functionRaw: .sigmoid)
            ]
            
            let set = Dataset(items: [
                DataItem(input: [0.0, 0.0, 0.0, 0.0], inputSize: .init(width: 2, height: 2), output: classifierOutput(classes: 2, correct: 1).body, outputSize: .init(width: 2)),
                DataItem(input: [0.0, 0.0, 0.0, 1.0], inputSize: .init(width: 2, height: 2), output: classifierOutput(classes: 2, correct: 0).body, outputSize: .init(width: 2)),
                DataItem(input: [0.0, 0.0, 1.0, 0.0], inputSize: .init(width: 2, height: 2), output: classifierOutput(classes: 2, correct: 1).body, outputSize: .init(width: 2)),
                DataItem(input: [0.0, 1.0, 0.0, 0.0], inputSize: .init(width: 2, height: 2), output: classifierOutput(classes: 2, correct: 1).body, outputSize: .init(width: 2)),
                DataItem(input: [1.0, 0.0, 0.0, 0.0], inputSize: .init(width: 2, height: 2), output: classifierOutput(classes: 2, correct: 1).body, outputSize: .init(width: 2)),
                DataItem(input: [1.0, 1.0, 1.0, 1.0], inputSize: .init(width: 2, height: 2), output: classifierOutput(classes: 2, correct: 0).body, outputSize: .init(width: 2))
            ])
            
            XCTAssertLessThan(network.train(set: set), 0.005)
        }
    }
