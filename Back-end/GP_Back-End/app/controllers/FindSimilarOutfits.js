var spawn = require("child_process").spawn; 
var sim_process=require("child_process").exec;
var multerUploadTemp = require('../multerUploadTemp');
var fs = require('fs');
exports.similarOutfits = function(req,res)
{
    multerUploadTemp(req, res,(error) => {
        if(error){
            console.log('invalid file type')
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
              
              // var process = spawn('python',["./findSimilarOutfit.py",imagePath] ); 

              cmd_string="cd C:\\Users\\mosama\\Anaconda3\\envs\\fash && python C:\\Users\\mosama\\PycharmProjects\\GP\\similar_outfits.py "+imagePath;
              // var process =sim_process(cmd_string);
              sim_process(cmd_string, (err, stdout, stderr) => {
                // console.log(stdout);
                // console.log(stdout.length);

                jsonSimilarOutfits = JSON.parse(stdout)
                console.log(jsonSimilarOutfits)
                for(var i = 0; i < jsonSimilarOutfits.length;i++)
                {
                  imagePath = jsonSimilarOutfits[i].path
                  var bitmap = fs.readFileSync(imagePath);
                  imageToBase64 = new Buffer(bitmap).toString('base64');
                  newImagePath = 'data:image/;base64,' + imageToBase64
                  jsonSimilarOutfits[i].path = newImagePath
                }
                  res.status(200).json({ similarOutfitsList: jsonSimilarOutfits });
                  return;

              });

           }
        }
      })
    } 
