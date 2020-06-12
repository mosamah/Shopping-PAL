
let NowTrending = require('../app/controllers/NowTrending');
let MatchOutfit = require('../app/controllers/MatchOutfit')
let FindSimilarPieces = require('../app/controllers/FindSimilarPieces')
let FindSimilarOutfits = require('../app/controllers/FindSimilarOutfits')
const express = require('express');
const router = express.Router();

//you can include all your controllers

router.get('/getDresses', NowTrending.getDresses)
router.get('/getJackets', NowTrending.getJackets)
router.get('/getPants', NowTrending.GetPants)
router.get('/getShirtsAndBlouses', NowTrending.GetShirtsAndBlouses)
router.get('/getSkirts', NowTrending.GetSkirts)
router.get('/getSweaters', NowTrending.GetSweaters)
router.get('/getTops', NowTrending.GetTops)
router.post('/segmentImage', MatchOutfit.processImage)
router.post('/matchOutfit', MatchOutfit.findMatches)
router.post('/segmentToPieces', FindSimilarPieces.pieceIdentification)
router.post('/findSimilarPieces',FindSimilarPieces.similarPieces)
router.post('/findSimilarOutfits', FindSimilarOutfits.similarOutfits)

module.exports = router;