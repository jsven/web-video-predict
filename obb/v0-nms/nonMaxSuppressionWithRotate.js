export function nonMaxSuppressionWithRotate(
	boxes,
	score,
	maxOutputSize,
	iouThreshold = 0.5,
	scoreThreshold = 0.3
) {
	// 初始化候选框数组
	let candidates = [];
	// 获取分数数组
	const scoreArray = score.arraySync();
	// 遍历分数数组，筛选出分数大于阈值的候选框
	for (let i = 0; i < scoreArray.length; i++) {
		if (scoreArray[i] > scoreThreshold) {
			candidates.push({ score: scoreArray[i], boxIndex: i, box: null });
		}
	}

	// 按分数从高到低排序候选框
	candidates.sort((a, b) => b.score - a.score);

	// 将候选框的索引转换为Tensor
	const candidatesTensor = tf.tensor1d(candidates.map(e => e.boxIndex), "int32");
	// 计算旋转后的矩阵
	const rotatedMatrix = rotationMatrix(boxes.gather(candidatesTensor, 0));
	// 将旋转后的矩阵转换为turf.js用的多边形
	const polygons = matrix2Polygons(rotatedMatrix);
	polygons.forEach((polygon, index) => {
		candidates[index].box = polygon;
	});

	// 初始化选择的索引数组
	const selectedIndices = [];

	// 当候选框还有剩余时，继续处理
	while (candidates.length > 0) {
		const currentCandidate = candidates[0];
		// 将当前候选框的索引添加到选择的索引数组中
		selectedIndices.push(currentCandidate.boxIndex);
		// 如果选择的索引数量达到最大输出大小，则停止
		if (selectedIndices.length >= maxOutputSize) {
			break;
		}
		// 过滤掉与当前候选框IoU大于阈值的候选框
		candidates = candidates.filter((candidate) => {
			if (candidate.boxIndex === currentCandidate.boxIndex) return false;
			const iou = calculateRotatedIOU(currentCandidate.box, candidate.box);
			return iou < iouThreshold;
		});
	}

	// 返回选择的索引数组作为Tensor
	return tf.tensor1d(selectedIndices, "int32");
}

// 计算两个旋转多边形的IoU
function calculateRotatedIOU(polygon_a, polygon_b) {
	// 计算两个多边形的交集
	const intersectPolygon = turf.intersect(turf.featureCollection([polygon_a, polygon_b]));
	if (!intersectPolygon) {
		return 0;
	}
	// 计算两个多边形的并集
	const unionPolygon = turf.union(turf.featureCollection([polygon_a, polygon_b]));
	if (!unionPolygon) {
		return 0;
	}
	// 计算IoU并返回
	const iou = turf.area(intersectPolygon) / turf.area(unionPolygon);
	return iou;
}

// 计算旋转后的矩阵
function rotationMatrix(boxes) {
	const results = tf.tidy(() => {
		// 分离出x, y, w, h, rad
		const [x, y, w, h, rad] = tf.split(boxes, [1, 1, 1, 1, 1], 1);
		// 计算cos和sin
		const cos = tf.cos(rad).squeeze();
		const sin = tf.sin(rad).squeeze();

		// 计算旋转后的四个点的坐标
		const x1 = w.div(-2).squeeze();
		const x2 = w.div(2).squeeze();
		const y1 = h.div(-2).squeeze();
		const y2 = h.div(2).squeeze();
		const p1x = x1.mul(cos).sub(y1.mul(sin)).add(x.squeeze());
		const p1y = x1.mul(sin).add(y1.mul(cos)).add(y.squeeze());
		const p2x = x2.mul(cos).sub(y1.mul(sin)).add(x.squeeze());
		const p2y = x2.mul(sin).add(y1.mul(cos)).add(y.squeeze());
		const p3x = x2.mul(cos).sub(y2.mul(sin)).add(x.squeeze());
		const p3y = x2.mul(sin).add(y2.mul(cos)).add(y.squeeze());
		const p4x = x1.mul(cos).sub(y2.mul(sin)).add(x.squeeze());
		const p4y = x1.mul(sin).add(y2.mul(cos)).add(y.squeeze());

		// 返回旋转后的四个点的坐标作为Tensor
		return tf.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y]);
	});
	return results;
}

// 将旋转后的矩阵转换为turf.js用的多边形
function matrix2Polygons(matrix) {
	const transposedMatrix = tf.tidy(() => {
		// 转置矩阵
		return matrix.transpose([1, 0]);
	});
	const matrixArray = transposedMatrix.arraySync();
	// 将矩阵数组转换为turf.js用的多边形
	return matrixArray.map(e => {
		return turf.polygon([[
			[e[0], e[1]],
			[e[2], e[3]],
			[e[4], e[5]],
			[e[6], e[7]],
			[e[0], e[1]],
		]]);
	});
}