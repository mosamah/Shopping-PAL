var mongoose = require('mongoose');
var Schema = mongoose.Schema;
var TopsSchema = new Schema({
  Title:  { type: String },
  Season: { type: String },
  Price:  { type: String },
  Fabric: { type: String },
  Style:  { type: String }
  }, { collection : 'shorts' }
  );
module.exports = mongoose.model('TopsModel', TopsSchema);