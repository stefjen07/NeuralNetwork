    import XCTest
    @testable import NeuralNetworkLibrary

    final class NeuralNetworkLibraryTests: XCTestCase {
        
        func testMatrixSum() throws {
            let matrix: [[Float]] = [
                [1,2],
                [3,4],
            ]
            let expected: Float = 10
            let result = matrixSum(matrix: matrix)
            XCTAssertEqual(result, expected)
        }
        
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
        
        func testOutNeuron() throws {
            let neuron = Neuron(weights: [0.2, 0.3], weightsDelta: [0.0, 0.0], bias: 0.0, delta: 0.0)
            let input: [Float] = [0.5, 0.3]
            let result = outNeuron(neuron, input: input)
            let expected = Float(0.19)
            XCTAssertEqual(result, expected)
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
            
            tempEnd.deltaWeights(input: dropout.deltaWeights(input: tempDense.deltaWeights(input: item.input, learningRate: learningRate), learningRate: learningRate), learningRate: learningRate)
            endDense.deltaWeights(input: dense.deltaWeights(input: item.input, learningRate: learningRate), learningRate: learningRate)
            
            tempEnd.updateWeights()
            endDense.updateWeights()
            
            XCTAssertEqual(tempEnd.neurons.map { $0.weights }, endDense.neurons.map { $0.weights })
        }
    }
