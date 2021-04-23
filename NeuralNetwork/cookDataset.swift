//
//  cookDataset.swift
//  NeuralNetwork
//
//  Created by Евгений on 20.04.2021.
//

import Foundation
import NeuralNetworkLibrary

func cook() {
    let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath + "/etl.ds")
    Dataset(folderPath: FileManager.default.currentDirectoryPath+"/Validation").save(to: url)
}

func getDS() -> Dataset {
    let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath + "/etl.ds")
    return Dataset(from: url)
}
