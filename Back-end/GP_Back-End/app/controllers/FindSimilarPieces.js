var spawn = require("child_process").spawn; 
var sim_process=require("child_process").exec;
var multerUploadTemp = require('../multerUploadTemp');
var fs = require('fs');

exports.pieceIdentification = function(req,res)
{
    multerUploadTemp(req, res,(error) => {
        if(error){
            console.log(error)
            res.send(500)   //500 means internal server error
            return;
         }else{
           if(req.file == undefined){
               console.log('file size too large')
               res.send(500);
               return;
          }else{

          var imagePath = "C:\\Users\\mosama\\PycharmProjects\\GP\\GP_Photos_Multer\\temp\\"+req.file.filename;
          console.log(imagePath)

          cmd_string="cd C:\\Users\\mosama\\Anaconda3\\envs\\fash && python C:\\Users\\mosama\\PycharmProjects\\GP\\piece_identification.py "+imagePath;
          sim_process(cmd_string, (err, stdout, stderr) => {
            console.log(stdout)
            jsonSegmentedImage = JSON.parse(stdout)
            console.log(jsonSegmentedImage.path)
            imagePath = jsonSegmentedImage.path
            var bitmap = fs.readFileSync(imagePath);
            imageToBase64 = new Buffer(bitmap).toString('base64');
            newImagePath = 'data:image/;base64,' + imageToBase64
            jsonSegmentedImage.path = newImagePath

            res.status(200).json({ segmentedImage: jsonSegmentedImage });
            return;

          })
         }
      }
    })
  } 






  exports.similarPieces = function(req,res)
  {
    imageLabel = req.body.chosenLabel;
    console.log(imageLabel)

    cmd_string="cd C:\\Users\\mosama\\Anaconda3\\envs\\fash && python C:\\Users\\mosama\\PycharmProjects\\GP\\get_similar_pieces.py "+imageLabel;

    sim_process(cmd_string, (err, stdout, stderr) => {

      jsonSimilarPieces = JSON.parse(stdout)
      console.log(jsonSimilarPieces)
      for(var i = 0; i < jsonSimilarPieces.length;i++)
      {
        imagePath = jsonSimilarPieces[i].path
        var bitmap = fs.readFileSync(imagePath);
        imageToBase64 = new Buffer(bitmap).toString('base64');
        newImagePath = 'data:image/;base64,' + imageToBase64
        jsonSimilarPieces[i].path = newImagePath
      }
      
      res.status(200).json({ similarPiecesList: jsonSimilarPieces });
      return;


    })
  } 
  
  
  



