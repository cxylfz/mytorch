#include<iostream>
#include<vector>
using namespace std;
struct block{
    int size;
    int addr;
};
vector<vector<block> > v(2000);
int max_block=1024;
pair<bool,int> allc(int need){
    int b_size=1;
    while(b_size<need){
        b_size*=2;
    }
    int now_size=b_size;
    while(now_size<=max_block){
        if(v[now_size].size()>0){
            int allo_addr=v[now_size][v[now_size].size()-1].addr;
            v[now_size].pop_back();
            while(now_size>b_size){
                now_size/=2;
                block new_block = {now_size,allo_addr};
                v[now_size].push_back(new_block);
                allo_addr+=now_size;
            }
            return pair<bool,int>(true,allo_addr);
        }
        now_size*=2;
    }
    return pair<bool,int>(false,-1);
}
void free(block b){
    int b_size=b.size;
    int now_addr=b.addr;
    while(b_size<max_block){
        int flag=0;
        for(int i=0;i<v[b_size].size();i++){
            if(now_addr+b_size==v[b_size][i].addr){
                if((now_addr^v[b_size][i].addr)==b_size){
                    v[b_size].erase(v[b_size].begin()+i);
                    b_size*=2;
                    flag=1;
                    break;
                }
            }
            if(now_addr-b_size==v[b_size][i].addr){
                if((now_addr^v[b_size][i].addr)==b_size){
                    v[b_size].erase(v[b_size].begin()+i);
                    now_addr-=b_size;
                    b_size*=2;
                    flag=1;
                    break;
                }
            }
        }
        if(flag==0){
            break;
        }
    }
    block new_block = {b_size,now_addr};
    v[new_block.size].push_back(new_block);
}
int main(){
    block father = {max_block,0};
    v[father.size].push_back(father);
    cout<<"请求分配128字节"<<endl;
    pair<bool,int>re=allc(128);
    if(re.first==true){
        cout<<"分配成功，起始地址为"<<re.second<<endl;
    }
    else{
        cout<<"分配失败"<<endl;
    }
    cout<<"请求分配32字节"<<endl;
    pair<bool,int>re1=allc(32);
    if(re1.first==true){
        cout<<"分配成功，起始地址为"<<re1.second<<endl;
    }
    else{
        cout<<"分配失败"<<endl;
    }
    cout<<"请求分配2048字节"<<endl;
    pair<bool,int>re2=allc(2048);
    if(re2.first==true){
        cout<<"分配成功，起始地址为"<<re2.second<<endl;
    }
    else{
        cout<<"分配失败"<<endl;
    }
    cout<<"当前内存状态为："<<endl;
    for(int i=1024;i>0;i/=2){
        cout<<i<<"字节内存有"<<v[i].size()<<"块："<<endl;
        for(int j=0;j<v[i].size();j++){
            cout<<"大小："<<v[i][j].size<<" 起始地址："<<v[i][j].addr<<endl;
        }
    }
    block block1={128,896};
    block block2={32,864};
    block block3={512,0};
    free(block2);
    free(block1);
    free(block3);
    cout<<"当前内存状态为："<<endl;
    for(int i=1024;i>0;i/=2){
        cout<<i<<"字节内存有"<<v[i].size()<<"块："<<endl;
        for(int j=0;j<v[i].size();j++){
            cout<<"大小："<<v[i][j].size<<" 起始地址："<<v[i][j].addr<<endl;
        }
    }
    return 0;
}