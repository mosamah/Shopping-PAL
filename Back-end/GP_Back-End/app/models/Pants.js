var mongoose = require('mongoose');
var Schema = mongoose.Schema;
var PantsSchema = new Schema({
  Title:  { type: String },
  Season: { type: String },
  Price:  { type: String },
  Fabric: { type: String },
  Style:  { type: String }
  }, { collection : 'pants' }
  );
module.exports = mongoose.model('PantsModel', PantsSchema);