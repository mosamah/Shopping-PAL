var mongoose = require('mongoose');
var Schema = mongoose.Schema;
var DressesSchema = new Schema({
  Title:  { type: String },
  Season: { type: String },
  Price:  { type: String },
  Fabric: { type: String },
  Style:  { type: String }
  }, { collection : 'dresses' }
  );
module.exports = mongoose.model('DressesModel', DressesSchema);