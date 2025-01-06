export function drawBbox(bboxes, context, imageSize, classNames) {
	// 清除context内容
	context.clearRect(0, 0, canvas.width, canvas.height)
	const scaleFactorWidth = canvas.width / imageSize[0]
	const scaleFactorHeight = canvas.height / imageSize[1]
	bboxes.forEach(bbox => {
		context.save()
		context.translate(
			bbox.x * scaleFactorWidth + ((bbox.w * scaleFactorWidth) / 2),
			bbox.y * scaleFactorHeight + ((bbox.h * scaleFactorHeight) / 2)
		)
		context.rotate(bbox.r)

		context.beginPath();
		context.rect(
			-bbox.w * scaleFactorWidth / 2,
			-bbox.h * scaleFactorHeight / 2,
			bbox.w * scaleFactorWidth,
			bbox.h * scaleFactorHeight)
		context.strokeStyle = "cyan"
		context.lineWidth = 1
		context.stroke()
		context.font = '16px Arial';
		context.fillStyle = 'green';
		context.fillText(
			classNames[bbox.label] ?? "none",
			-bbox.w * scaleFactorWidth / 2,
			-bbox.h * scaleFactorHeight / 2)

		context.restore()
	})
}