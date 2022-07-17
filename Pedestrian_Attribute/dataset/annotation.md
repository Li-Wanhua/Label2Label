### dataset split
  - PETA 19000,train 9500,val 1900, test 7600
  - RAP 41585,train 33268, test 8317
  - PA-100K 100000,train 80000, val 10000, test 10000
  - RAPv2 84928, train, 50957 trainset, 16986 valset, 16985 testset



|type|dataset|attribute|image|train|val|test|
|----|-------|---------|-----|-----|---|----|
|pedestrian|PETA|35|19,000|9,500|1,900|7,600|
|          |RAP|54|84,928|50,957|16,986|16,985|
|          |PA100k|26|100,000|80,000|10,000|10,000|
|fashion|DeepFashion|1,000|289222|209,252|40,052|40,000|
|       |iFashion|228|1,062,550|1,012,947|9,897|39,706|
|face|CelebA|40|200,000|160,000|20,000|20,000|
|general|pascal voc|20|9967|5011|--|4952|
|       |MS-COCO|80|122,218|82,081|--|40,137|
|       |NUS-WIDE|81|168635|100,893|--|67,742|



## 统一属性顺序
1. head region
2. upper region
3. lower region
4. foot region
5. accessory/bag
6. age
7. gender
8. others

### WIDER attribute

num_ingroup = [4, 6, 3, 1]

- longHair, sunglass, Hat, faceMask [1, 2, 3, 11] 4
- Tshirt, longSleeve, formal, shorts, logo, stripe [4, 5, 6, 7, 12, 13] 6
- jeans, longPants, skirt [8, 9, 10] 3
- Male [0] 1

permutation = [1, 2, 3, 11, 4, 5, 6, 7, 12, 13, 8, 9, 10, 0]


### PETA （35 in 105）
num_ingroup = [5, 10, 6, 4, 5, 5]

- 'accessoryHat','accessoryMuffler','accessoryNothing','accessorySunglasses','hairLong' [10, 18, 19, 30, 15] 5
- 'upperBodyCasual', 'upperBodyFormal', 'upperBodyJacket', 'upperBodyLogo', 'upperBodyPlaid', 'upperBodyShortSleeve', 'upperBodyThinStripes', 'upperBodyTshirt','upperBodyOther','upperBodyVNeck'  [7, 9, 11, 14, 21, 26, 29, 32, 33, 34] 10
- 'lowerBodyCasual', 'lowerBodyFormal', 'lowerBodyJeans', 'lowerBodyShorts', 'lowerBodyShortSkirt','lowerBodyTrousers' [6, 8, 12, 25, 27, 31] 6
- 'footwearLeatherShoes', 'footwearSandals', 'footwearShoes', 'footwearSneaker' [13, 23, 24, 28] 4
- 'carryingBackpack', 'carryingOther', 'carryingMessengerBag', 'carryingNothing', 'carryingPlasticBags' [4, 5, 17, 20, 22] 5

- 'personalLess30','personalLess45','personalLess60','personalLarger60', [0, 1, 2, 3] 4
- 'personalMale', [16] 1

permutation = [10, 18, 19, 30, 15, 7, 9, 11, 14, 21, 26, 29, 32, 33, 34, 6, 8, 12, 25, 27, 31, 13, 23, 24, 28, 4, 5, 17, 20, 22, 0, 1, 2, 3, 16]
 
 
### PA100K （26)
num_ingroup = [2, 6, 6, 1, 4, 7]

- 'Hat','Glasses', [7,8] 2
- 'ShortSleeve','LongSleeve','UpperStride','UpperLogo','UpperPlaid','UpperSplice', [13,14,15,16,17,18] 6
- 'LowerStripe','LowerPattern','LongCoat','Trousers','Shorts','Skirt&Dress', [19,20,21,22,23,24] 6
- 'boots' [25] 1
- 'HandBag','ShoulderBag','Backpack','HoldObjectsInFront', [9,10,11,12] 4

- 'AgeOver60','Age18-60','AgeLess18', [1,2,3] 3
- 'Female' [0] 1
- 'Front','Side','Back', [4,5,6] 3

permutation = [7,8,13,14,15,16,17,18,19,20,21,22,23,24,25,9,10,11,12,1,2,3,0,4,5,6]


'Hat',
'Glasses'
'ShortSleeve','LongSleeve'
'AgeOver60','Age18-60','AgeLess18', 
'Female'
'Front','Side','Back', 
'boots' 

'UpperStride','UpperLogo','UpperPlaid','UpperSplice', 
'LowerStripe','LowerPattern','LongCoat','Trousers','Shorts','Skirt&Dress', [19,20,21,22,23,24] 6

'HandBag','ShoulderBag','Backpack','HoldObjectsInFront', 






### RAPv1 (51)

num_ingroup = [6, 9, 6, 5, 8, 17]

- head 6：'hs-BaldHead','hs-LongHair','hs-BlackHair','hs-Hat','hs-Glasses','hs-Muffler', [9, 10, 11, 12, 13, 14,]
- upper body 9：'ub-Shirt','ub-Sweater','ub-Vest','ub-TShirt','ub-Cotton','ub-Jacket','ub-SuitUp','ub-Tight','ub-ShortSleeve',[15, 16, 17, 18, 19, 20, 21, 22, 23,]
- lower body 6：'lb-LongTrousers','lb-Skirt','lb-ShortSkirt','lb-Dress','lb-Jeans','lb-TightTrousers', [24, 25,26, 27, 28, 29,]
- footwear 5：'shoes-Leather','shoes-Sport','shoes-Boots','shoes-Cloth','shoes-Casual', [30, 31, 32, 33, 34,]
- accessory 8 [35, 36, 37, 38, 39, 40, 41, 42] :
'attach-Backpack','attach-SingleShoulderBag','attach-HandBag','attach-Box','attach-PlasticBag','attach-PaperBag','attach-HandTrunk','attach-Other',

- age 3：'AgeLess16','Age17-30','Age31-45', 1:4 [1, 2, 3,]
- gender 1：'Female', 0 [0,]
- body shape 3：'BodyFat','BodyNormal','BodyThin',4:7 [4, 5, 6,]
- role 2：'Customer','Clerk', 7:9 [ 7, 8,]
- action 8：'action-Calling','action-Talking','action-Gathering','action-Holding','action-Pusing','action-Pulling','action-CarrybyArm','action-CarrybyHand'
[43, 44, 45, 46, 47, 48, 49, 50]

permutation = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
       26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 1, 2, 3, 0, 4, 5, 6, 7, 8,  43, 44, 45, 46,
       47, 48, 49, 50]
       
### RAPv2 (54)
num_ingroup = [5, 10, 6, 6, 8, 19]

- head 5：'hs-BaldHead', 'hs-LongHair', 'hs-BlackHair', 'hs-Hat', 'hs-Glasses', [10,11,12,13,14]
- upper body 10：'ub-Shirt','ub-Sweater','ub-Vest','ub-TShirt','ub-Cotton','ub-Jacket','ub-SuitUp','ub-Tight','ub-ShortSleeve','ub-Others'
[15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
- lower body 6：'lb-LongTrousers','lb-Skirt','lb-ShortSkirt','lb-Dress','lb-Jeans','lb-TightTrousers', [25 ,26, 27, 28, 29, 30]
- footwear 6：'shoes-Leather', 'shoes-Sports', 'shoes-Boots', 'shoes-Cloth', 'shoes-Casual', 'shoes-Other', [31, 32, 33, 34, 35, 36]
- accessory 8 [37, 38, 39, 40, 41, 42, 43, 44] :
'attachment-Backpack','attachment-ShoulderBag','attachment-HandBag','attachment-Box','attachment-PlasticBag','attachment-PaperBag','attachment-HandTrunk','attachment-Other'

- age 4：'AgeLess16', 'Age17-30', 'Age31-45', 'Age46-60', [1, 2, 3, 4]
- gender 1：'Female', [0,]
- body shape 3：'BodyFat','BodyNormal','BodyThin',4:7 [5, 6, 7]
- role 2：'Customer','Employee', [ 8, 9,]
- action 9：'action-Calling','action-Talking','action-Gathering','action-Holding','action-Pushing','action-Pulling','action-CarryingByArm','action-CarryingByHand','action-Other'
[45, 46, 47, 48, 49, 50, 51, 52, 53]
       
permutation = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
               36, 37, 38, 39, 40, 41, 42, 43, 44, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 45, 46, 47, 48, 49, 50, 51, 52, 53]
