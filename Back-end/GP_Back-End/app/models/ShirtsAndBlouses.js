var mongoose = require('mongoose');
var Schema = mongoose.Schema;
var ShirtsAndBlousesSchema = new Schema({
  Title:  { type: String },
  Season: { type: String },
  Price:  { type: String },
  Fabric: { type: String },
  Style:  { type: String }
  }, { collection : 'blouses' }
  );
module.exports = mongoose.model('ShirtsAndBlousesModel', ShirtsAndBlousesSchema);