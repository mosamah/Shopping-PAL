var mongoose = require('mongoose');
var Schema = mongoose.Schema;
var SweatersSchema = new Schema({
  Title:  { type: String },
  Season: { type: String },
  Price:  { type: String },
  Fabric: { type: String },
  Style:  { type: String }
  }, { collection : 'sweaters' }
  );
module.exports = mongoose.model('SweatersModel', SweatersSchema);