//
//  BoundingBoxView.swift
//  defectdettest
//
//  Created by Shubham  on 04/06/25.
//

import SwiftUI

struct BoundingBoxView: View {
    let boundingBox: CGRect
    let imageSize: CGSize

    var body: some View {
        let rect = CGRect(
            x: boundingBox.origin.x * imageSize.width,
            y: (1 - boundingBox.origin.y - boundingBox.height) * imageSize.height,
            width: boundingBox.width * imageSize.width,
            height: boundingBox.height * imageSize.height
        )

        return Rectangle()
            .stroke(Color.red, lineWidth: 2)
            .frame(width: rect.width, height: rect.height)
            .position(x: rect.midX, y: rect.midY)
    }
}


