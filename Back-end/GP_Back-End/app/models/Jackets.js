var mongoose = require('mongoose');
var Schema = mongoose.Schema;
var JacketsSchema = new Schema({
  path:  { type: String },
  labels: { type: Array }
  }, { collection : 'dataset' }
  );
module.exports = mongoose.model('JacketsModel', JacketsSchema);