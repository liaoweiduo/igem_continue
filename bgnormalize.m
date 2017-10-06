function [imbn, imer] = bgnormalize(imod)
% Background Normalize has three steps:
%   Denoise
%   Background normalize
%   Extend SNR ratio
%
% [imbn, imer] = bgnormalize(imod) returns
%   normalized and extended ratio image.

% Convert to double
if ~isfloat(imod(1))
    imod = double(imod);
end

% Denoise
impz = padarray(imod, [8 8], 'symmetric');
imwf = wiener2(impz, [9 9]);

% Background normalize
% Rolling ball structure with R = 50, H = 50
se = strel('ball', 50, 50);
imwf(imwf < 100) = 100;
imop = imopen(imwf, se);
imwf = imwf(9:end-8, 9:end-8);
imop = imop(9:end-8, 9:end-8);
imbn = imwf - imop;     % imtophat(imwf, se)

% Extend SNR ratio
imrt = imwf ./ imop;    % SNR ratio
imer = imrt .* imbn;    % Extend ratio

% Background flag
bgf = imbn<1 | imrt<1.2;
bg = mean(imbn(bgf)) + 3*std(imbn(bgf));
bgf = bgf | imbn<bg;
imer(bgf) = 0;
imer = medfilt2(imer);  % Smooth objects' edge

