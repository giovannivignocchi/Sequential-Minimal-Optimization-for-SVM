function data = corners(N, base, height, gapwidth)

    if nargin < 1
        N = 1000;
    end
    if mod(N,4) ~= 0
        N = round(N/4) * 4;
    end

    if nargin < 2
        base = 10;
    end
    if nargin < 3
        height = 10;
    end   
    if nargin < 4
        gapwidth = 1;
    end

    perCorner = N/4;
    
    NE = [ - base / 2 + (base/2 - gapwidth) .* rand(perCorner,1) ,   height / 2 - (height/2 - gapwidth) .* rand(perCorner,1) , ones(perCorner,1)];
    NO = [   base / 2 - (base/2 - gapwidth) .* rand(perCorner,1) ,   height / 2 - (height/2 - gapwidth) .* rand(perCorner,1) , -ones(perCorner,1)];
    SE = [ - base / 2 + (base/2 - gapwidth) .* rand(perCorner,1) , - height / 2 + (height/2 - gapwidth) .* rand(perCorner,1) , -ones(perCorner,1)];
    SO = [   base / 2 - (base/2 - gapwidth) .* rand(perCorner,1) , - height / 2 + (height/2 - gapwidth) .* rand(perCorner,1) , ones(perCorner,1)];
    
    data=  [NE; NO; SE; SO];

end