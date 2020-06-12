let DressesModel = require('../models/Dresses');
let JacketsModel = require('../models/Jackets');
let PantsModel = require('../models/Pants');
let ShirtsAndBlousesModel = require('../models/ShirtsAndBlouses');
let SkirtsModel = require('../models/Skirts');
let SweatersModel = require('../models/Sweaters');
let TopsModel = require('../models/Tops');
var fs = require('fs');

exports.getDresses = function(req, res)
{
  DressesModel.find({}, ['_id','Title','Season','Price','Fabric',
    'Style'], {sort:{ _id: -1} }, function(err, Dresses) {
    for(var i = 0; i < Dresses.length;i++)
    {
      console.log(Dresses[i])
      titleName=Dresses[i].Title
      imagePath = 'C:\\Users\\mosama\\PycharmProjects\\GP\\crawled_images\\Shein Images\\Shein Crawled Dresses\\'+titleName+'.jpg'
      var bitmap = fs.readFileSync(imagePath);
      imageToBase64 = new Buffer(bitmap).toString('base64');
      newImagePath = 'data:image/;base64,' + imageToBase64
      Dresses[i].Title = newImagePath

    }
    if(err) throw err;
    //res.render('index', { title: 'NodeJS file upload ', msg:req.query.msg, photolist : photos });
    res.status(200).json({ DressesList: Dresses });
   }).limit(50);

}

exports.getJackets = function(req, res)
{
	JacketsModel.find({}, ['_id','path','labels'], {sort:{ _id: -1} }, function(err, Jackets) {
    for(var i = 0; i < Jackets.length;i++)
    {
      console.log(Jackets[i])
      imagePath = Jackets[i].path
      var bitmap = fs.readFileSync(imagePath);
      imageToBase64 = new Buffer(bitmap).toString('base64');
      newImagePath = 'data:image/;base64,' + imageToBase64
      Jackets[i].path = newImagePath
    }
    if(err) throw err;
    //res.render('index', { title: 'NodeJS file upload ', msg:req.query.msg, photolist : photos });
    res.status(200).json({ JacketsList: Jackets });
   }).limit(50);

}

exports.GetPants = function(req, res)
{
	PantsModel.find({}, ['_id','Title','Season','Price','Fabric',
  'Style'], {sort:{ _id: -1} }, function(err, Pants) {
    for(var i = 0; i < Pants.length;i++)
    {

      console.log(Pants[i])
      titleName=Pants[i].Title
      imagePath = 'C:\\Users\\mosama\\PycharmProjects\\GP\\crawled_images\\Shein Images\\Shein Crawled Pants\\'+titleName+'.jpg'
      var bitmap = fs.readFileSync(imagePath);
      imageToBase64 = new Buffer(bitmap).toString('base64');
      newImagePath = 'data:image/;base64,' + imageToBase64
      Pants[i].Title = newImagePath

    }
    if(err) throw err;
    //res.render('index', { title: 'NodeJS file upload ', msg:req.query.msg, photolist : photos });
    res.status(200).json({ PantsList: Pants });
   }).limit(50);

}

exports.GetShirtsAndBlouses = function(req, res)
{
	ShirtsAndBlousesModel.find({}, ['_id','Title','Season','Price','Fabric',
  'Style'], {sort:{ _id: -1} }, function(err, ShirtsAndBlouses) {
    for(var i = 0; i < ShirtsAndBlouses.length;i++)
    {
      console.log(ShirtsAndBlouses[i])
      titleName=ShirtsAndBlouses[i].Title
      imagePath = 'C:\\Users\\mosama\\PycharmProjects\\GP\\crawled_images\\Shein Images\\Shein Crawled Blouses\\'+titleName+'.jpg'
      // imagePath = ShirtsAndBlouses[i].path
      var bitmap = fs.readFileSync(imagePath);
      imageToBase64 = new Buffer(bitmap).toString('base64');
      newImagePath = 'data:image/;base64,' + imageToBase64
      ShirtsAndBlouses[i].Title = newImagePath

    }
    if(err) throw err;
    //res.render('index', { title: 'NodeJS file upload ', msg:req.query.msg, photolist : photos });
    res.status(200).json({ ShirtsAndBlousesList: ShirtsAndBlouses });
   }).limit(20);

}

exports.GetSkirts = function(req, res)
{
	SkirtsModel.find({}, ['_id','Title','Season','Price','Fabric',
  'Style'], {sort:{ _id: -1} }, function(err, Skirts) {
    for(var i = 0; i < Skirts.length;i++)
    {
      console.log(Skirts[i])
      titleName=Skirts[i].Title
      imagePath = 'C:\\Users\\mosama\\PycharmProjects\\GP\\crawled_images\\Shein Images\\Shein Crawled Skirts\\'+titleName+'.jpg'
      // imagePath = Skirts[i].path
      var bitmap = fs.readFileSync(imagePath);
      imageToBase64 = new Buffer(bitmap).toString('base64');
      newImagePath = 'data:image/;base64,' + imageToBase64
      Skirts[i].Title = newImagePath

    }
    if(err) throw err;
    //res.render('index', { title: 'NodeJS file upload ', msg:req.query.msg, photolist : photos });
    res.status(200).json({ SkirtsList: Skirts });
   }).limit(50);

}

exports.GetSweaters = function(req, res)
{
	SweatersModel.find({}, ['_id','Title','Season','Price','Fabric',
  'Style'], {sort:{ _id: -1} }, function(err, Sweaters) {
    for(var i = 0; i < Sweaters.length;i++)
    {
      console.log(Sweaters[i])
      titleName=Sweaters[i].Title
      imagePath = 'C:\\Users\\mosama\\PycharmProjects\\GP\\crawled_images\\Shein Images\\Shein Crawled Sweaters\\'+titleName+'.jpg'
      // imagePath = Sweaters[i].path
      var bitmap = fs.readFileSync(imagePath);
      imageToBase64 = new Buffer(bitmap).toString('base64');
      newImagePath = 'data:image/;base64,' + imageToBase64
      Sweaters[i].Title = newImagePath

    }
    if(err) throw err;
    //res.render('index', { title: 'NodeJS file upload ', msg:req.query.msg, photolist : photos });
    res.status(200).json({ SweatersList: Sweaters });
   }).limit(50);

}

exports.GetTops = function(req, res)
{
	TopsModel.find({}, ['_id','Title','Season','Price','Fabric',
  'Style'], {sort:{ _id: -1} }, function(err, Tops) {
    console.log(Tops)
    for(var i = 0; i < Tops.length;i++)
    {
      console.log(Tops[i])
      titleName=Tops[i].Title
      imagePath = 'C:\\Users\\mosama\\PycharmProjects\\GP\\crawled_images\\Shein Images\\Shein Crawled Shorts\\'+titleName+'.jpg'
      // imagePath = Tops[i].path

      var bitmap = fs.readFileSync(imagePath);
      imageToBase64 = new Buffer(bitmap).toString('base64');
      newImagePath = 'data:image/;base64,' + imageToBase64
      Tops[i].Title = newImagePath

    }
    if(err) throw err;
    //res.render('index', { title: 'NodeJS file upload ', msg:req.query.msg, photolist : photos });
    res.status(200).json({ TopsList: Tops });
   }).limit(50);

}