
#include "beauty/commontools.h"


using namespace std;

bool TimeStatic(int id,const char* type)
{ 

    struct TimeInfo
    {
        long long accu_num;
        long long accu_sec;
        long long accu_usec;

        struct timeval st;
        struct timeval ed;
        long long this_time_usec;

        char type[64];
    };
    static TimeInfo info[50];
    
    if(id<0)
    {
        for(int i=0;i<50;i++)memset(info+i,0,sizeof(TimeInfo));
        return true;
    }

    if(type==NULL)
    {
        gettimeofday(&info[id].st,NULL);
        return true;
    }
    gettimeofday(&info[id].ed,NULL);
    info[id].this_time_usec=((info[id].ed.tv_sec)-(info[id].st.tv_sec))*1000000 +
                ((info[id].ed.tv_usec)-(info[id].st.tv_usec));  

    if(info[id].type[0]=='\0') strcpy(info[id].type,type);
    bool needPrint=false;   
    info[id].accu_num++;
    info[id].accu_usec+=info[id].this_time_usec;
 
    char typeData[100];
    sprintf(typeData,"%d-%s",id,info[id].type);

    char tmp[256];
    sprintf(tmp,"=========step: %s, this time: %lld ms=========",typeData,info[id].this_time_usec / 1000);
    printf("%s\n",tmp);
    return true;
}