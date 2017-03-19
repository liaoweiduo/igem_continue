function [ OutImg ] = DivBackground( InImg , BgX , BgY, BgLength , BgWidth )
    BgImg  = InImg(BgY:BgY+BgWidth,BgX:BgX+BgLength);
    BgImg_t = BgImg(:);
    BgImg_t = sort(BgImg_t,'descend');
    BgInt = mean(BgImg_t( : ));

    OutImg = InImg-BgInt;
end

