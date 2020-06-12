var mongoose = require('mongoose');
var Schema = mongoose.Schema;
var SkirtsSchema = new Schema({
  Title:  { type: String },
  Season: { type: String },
  Price:  { type: String },
  Fabric: { type: String },
  Style:  { type: String }
  }, { collection : 'skirts' }
  );
module.exports = mongoose.model('SkirtsModel', SkirtsSchema);