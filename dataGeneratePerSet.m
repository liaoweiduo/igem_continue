%parameter initial
clear all;
rawpath = {...
    '/Volumes/Seagate BUP/IGEM_new/20170318/5ul/piezo+5ult%03dc1.tif',...
    };
nf=size(rawpath,2);  %num of files
nt=500;%num of pictures in one file
np=6;  %num of points in one picture
interval=0.4;
skipRate=10; %picture flash rate
caxis_left=300;caxi_right2=1400;
Title = '20170318,piezo,5ul';
Legend= {...
    'area 1',...
    'area 2',...
    'area 3',...
    'area 4',...
    'area 5',...
    'area 6',...
    };
co={'r','g','b','c','m','y'};
Int = zeros(nf,nt,np);X = zeros(nf,np);Y = zeros(nf,np);L = zeros(nf,np);W = zeros(nf,np);BgX = zeros(nf);BgY = zeros(nf);BgLength = zeros(nf);BgWidth = zeros(nf);

figure(1);
set (gcf,'Position',[0,0,1600,900]);
subplot(1,2,1);

for i=1:nf
    isSkip = 0;
    for i1=1:skipRate:nt
        if mod(i1,20)==0
            fprintf('the %d file: %d pictures\n',i,i1);
        end
        rawfile=sprintf(rawpath{i}, i1); rawImg = importdata(rawfile); rawImg=rawImg(:,:,1);
        imagesc(rawImg);colorbar;
        caxis([caxis_left,caxi_right2]);
        t = title(strcat('pause key: ',char(int16('a')+i-1)));set(t,'fontsize',20);
        if strcmpi(get(gcf,'CurrentCharacter'),char(int16('a')+i-1))
            break;
        elseif strcmpi(get(gcf,'CurrentCharacter'),char(int16('1')+i-1))
            isSkip=1;
            break;
        end
        pause(0.05);
    end
    if isSkip==1
        BgX(i)=uint32(BgX(i-1));BgY(i)=uint32(BgY(i-1));BgLength(i)=uint32(BgLength(i-1));BgWidth(i)=uint32(BgWidth(i-1));
        for ip=1:np
            X(i,ip)=uint32(X(i-1,ip));Y(i,ip)=uint32(Y(i-1,ip));L(i,ip)=uint32(L(i-1,ip));W(i,ip)=uint32(W(i-1,ip));
        end
        continue;
    end
    t = title('Picture');set(t,'fontsize',20);
    [x, y]=ginput(2);BgX(i)=uint32(x(1));BgY(i)=uint32(y(1));BgLength(i)=uint32(x(2))-uint32(x(1));BgWidth(i)=uint32(y(2))-uint32(y(1));
    rectangle('Position',[BgX(i),BgY(i),BgLength(i),BgWidth(i)],'EdgeColor','k');
    pause(1);

    for ip=1:np
        [x, y]=ginput(2);X(i,ip)=uint32(x(1));Y(i,ip)=uint32(y(1));L(i,ip)=uint32(x(2))-uint32(x(1));W(i,ip)=uint32(y(2))-uint32(y(1));
        rectangle('Position',[X(i,ip),Y(i,ip),L(i,ip),W(i,ip)],'EdgeColor',co{ip});
        pause(1);
    end
end

for i=1:nf
    parfor i1=1:nt
        if mod(i1,20)==0
            fprintf('the %d file: %d pictures\n',i,i1);
        end
        rawfile=sprintf(rawpath{i}, i1); rawImg = importdata(rawfile); rawImg=rawImg(:,:,1);
        lowNoisyImg = DivBackground(rawImg, uint32(BgX(i)), uint32(BgY(i)), uint32(BgLength(i)), uint32(BgWidth(i)));
        for ip=1:np
            LNImg = lowNoisyImg(Y(i,ip):Y(i,ip)+W(i,ip),X(i,ip):X(i,ip)+L(i,ip));
            LNImg_t = LNImg(:);LNImg_t = sort(LNImg_t,'descend');
            Int(i,i1,ip) = mean(LNImg_t(uint32(L(i,ip)*W(i,ip)*0.01) : uint32(L(i,ip)*W(i,ip)*0.11)));
        end
    end
    for i1=2:nt
        for ip=1:np
            if Int(i,1,ip)==0
                Int(i,i1,ip)=0;
            else
                Int(i,i1,ip)=(Int(i,i1,ip)-Int(i,1,ip))./Int(i,1,ip);
            end
        end
    end
    for ip=1:np
        Int(i,1,ip)=0;
    end
end

Int=permute(Int,[3,2,1]);

%figure
subplot(1,2,2);
hold on;
bMatrix=(1:nt);
for i=1:size(Int,1)
    Intf=permute(Int(i,:,:),[2,3,1])';
    Intfm=mean(Intf,1);Intfm=Intfm(bMatrix);
    %Intfs=std(Intf);Intfs=Intfs(bMatrix);
    %errorbar(bMatrix.*interval,Intfm,Intfs,'-o');
    plot(bMatrix.*interval,Intfm,co{i});
end
h = legend(Legend);set(h,'fontsize',20);
t = title(Title);set(t,'fontsize',20);
xlabel('Time(s)','fontsize',15);
ylabel('Fluorescent Intensity, dF/F0','fontsize',15);
hold off;





