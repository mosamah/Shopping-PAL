var multer = require('multer');
var path   = require('path');

/** Storage Engine */
const storageEngine = multer.diskStorage({
    destination: 'C:\\Users\\mosama\\PycharmProjects\\GP\\GP_Photos_Multer\\temp',
    filename: function(req, file, fn){
      fn(null,  new Date().getTime().toString()+'-'+file.fieldname+path.extname(file.originalname));
    }
  }); 
  
  //init
  
  const multerUploadTemp =  multer({
    storage: storageEngine,
    limits: { fileSize:10000000 },
    fileFilter: function(req, file, callback){
      validateFile(file, callback);
    }
  }).single('photo');
  
  
  var validateFile = function(file, cb ){
    allowedFileTypes = /jpeg|jpg|png/;
    const extension = allowedFileTypes.test(path.extname(file.originalname).toLowerCase());
    const mimeType  = allowedFileTypes.test(file.mimetype);
    if(extension && mimeType){
      return cb(null, true);
    }else{
      cb("Invalid file type. Only JPEG, PNG and GIF file are allowed.")
    }
  }

  module.exports = multerUploadTemp;
  