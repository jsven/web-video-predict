export default class ObbPredict {
  constructor(tf, modelJsonUrl, _YoloMetaData) {
    this._tf_ = tf;
    this.modelJsonUrl = modelJsonUrl;
    this.YoloMetaData = {
      ...this.YoloMetaData,
      _YoloMetaData
    }
  }

  model = null;
  _tf_ = null;
  YoloMetaData = {
    description: '',
    author: '',
    date: '',
    version: '',
    license: '',
    docs: '',
    stride: '',
    task: '',
    batch: 1,
    imgSize: [640, 640],
    names: [],
  };
  modelJsonUrl = '';

  async loadYOLOModel() {
    const model = await this._tf_.loadGraphModel(this.modelJsonUrl);
    // warm up
    this._tf_.tidy(() => {
      const zeroTensor = this._tf_.zeros(
        [1, 3, this.YoloMetaData.imgSize[0], this.YoloMetaData.imgSize[1]],
        'float32'
      );
      model.execute(zeroTensor);
    });
    this.model = model;
  }

  async predict(img) {
    const imageTensor = this._tf_.tidy(() => {
      let imageTensor = this._tf_.browser
        .fromPixels(img)
        .toFloat()
        .div(this._tf_.scalar(255.0));
      const resizedImageTensor = imageTensor.resizeBilinear(
        this.YoloMetaData.imgSize
      );
      // 调整维度顺序并添加批次维度
      const adjustedImageTensor = resizedImageTensor
        .transpose([2, 0, 1])
        .reshape([
          1,
          3,
          this.YoloMetaData.imgSize[0],
          this.YoloMetaData.imgSize[1],
        ]);
      // 确保数据类型为 float32
      const finalImageTensor = adjustedImageTensor.toFloat();
      return finalImageTensor;
    });

    const results = await this.model.execute(imageTensor);

    return results;
  }
}
