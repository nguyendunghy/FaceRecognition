for i = 1:10
   in = strcat(int2str(i),'.pgm');
   out = strcat(int2str(i),'.jpg');
   im = imread(in);
   imwrite(im,out);
end