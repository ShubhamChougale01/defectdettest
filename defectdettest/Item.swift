//
//  Item.swift
//  defectdettest
//
//  Created by Shubham  on 04/06/25.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
