%parameter initial
clear all;
rawpath = '/Volumes/Seagate BUP/IGEM_new/20170409/GECO/geco1ul/geco1ult150c1.tif';
rawImg = importdata(rawpath); rawImg=rawImg(:,:,1);
[imbn, imer] = bgnormalize(rawImg);
imwrite( uint8(imer),'/Volumes/Seagate BUP/IGEM_new/20170409/GECO/geco1ul/geco1ul.tif','tif' );
