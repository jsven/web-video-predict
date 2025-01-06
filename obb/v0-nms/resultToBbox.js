import {nonMaxSuppressionWithRotate} from './nonMaxSuppressionWithRotate.js';

// 将模型的输出结果转换为边界框
export const resultToBbox = (result, labelCount) => {
  const bboxes = tf.tidy(() => {
    const temp = result.squeeze();
    // 从模型输出中提取x, y, w, h, r
    const x = temp.slice([0, 0], [1, -1]); // x坐标
    const y = temp.slice([1, 0], [1, -1]); // y坐标
    const w = temp.slice([2, 0], [1, -1]); // 宽度
    const h = temp.slice([3, 0], [1, -1]); // 高度
    const r = temp.slice([(result.shape[1] ?? 0) - 1, 0], [1, -1]); // 旋转角度

    // 计算边界框的左上角和右下角坐标
    const x1 = tf.sub(x, tf.div(w, 2));
    const y1 = tf.sub(y, tf.div(h, 2));
    const x2 = tf.add(x1, w);
    const y2 = tf.add(y1, h);
    const boxes = tf.stack([y1, x1, y2, x2], 2).squeeze();
    const boxesWithR = tf.stack([x, y, w, h, r], 2).squeeze();

    // 提取每个边界框的最大分数和对应的标签索引
    const maxScores = temp.slice([4, 0], [labelCount, -1]).max(0);
    const labelIndexes = temp.slice([4, 0], [labelCount, -1]).argMax(0);
    const bboxIndexs = nonMaxSuppressionWithRotate(
      boxesWithR.as2D(boxesWithR.shape[0], boxesWithR.shape[1]),
      maxScores.as1D(),
      200,
      0.45,
      0.4
    );

    // 提取非最大值抑制后的边界框、分数、标签和旋转角度
    const resultBboxes = boxes.gather(bboxIndexs, 0).arraySync();
    const resultScores = maxScores.gather(bboxIndexs, 0).arraySync();
    const resultLables = labelIndexes.gather(bboxIndexs, 0).arraySync();
    const rs = r.squeeze().gather(bboxIndexs, 0).arraySync();

    // 返回边界框的信息对象
    return resultBboxes.map((bbox, index) => {
      return {
        x: bbox[1],
        y: bbox[0],
        w: bbox[3] - bbox[1],
        h: bbox[2] - bbox[0],
        score: resultScores[index],
        label: resultLables[index],
        r: rs[index]
      };
    });
  });
  return bboxes;
};