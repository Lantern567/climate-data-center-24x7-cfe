#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <array>

using Grad=std::array<double,3>;
static Grad sub(Grad a,Grad b){for(int i=0;i<3;i++)a[i]-=b[i];return a;}

// Exact separating cuts for wind MW, PV MW and storage MWh. Each weather
// year's energy balance, worst cyclic energy drawdown, and AC discharge power
// are concave feasibility functions. Their supergradients yield necessary LP
// cuts; feasibility plus the LP lower bound certifies the resulting optimum.
extern "C" void gf_cuts(const float* d,const float* pv,const float* wind,
 const long long* ends,int ny,double p,double w,double e,double duration,
 double eta,double* out) {
    const double leg=std::sqrt(eta);
    for(int y=0;y<ny;y++) {
        double cumulative=0,hi=0,lo=0,fall=0,rise=0,power_min=1e100;
        Grad cg={0,0,0},hg={0,0,0},lg={0,0,0},fg={0,0,0},rg={0,0,0},pg={0,0,0};
        for(long long t=ends[y];t<ends[y+1];t++) {
            const double net=p*pv[t]+w*wind[t]-d[t];
            const double headroom=net+e/duration;
            if(headroom<power_min){power_min=headroom;pg={pv[t],wind[t],1/duration};}
            double increment;Grad g;
            if(net<0){increment=net/leg;g={pv[t]/leg,wind[t]/leg,0};}
            else if(net<e/duration){increment=net*leg;g={pv[t]*leg,wind[t]*leg,0};}
            else {increment=e*leg/duration;g={0,0,leg/duration};}
            cumulative+=increment;for(int j=0;j<3;j++)cg[j]+=g[j];
            if(hi-cumulative>fall){fall=hi-cumulative;fg=sub(hg,cg);}
            if(cumulative-lo>rise){rise=cumulative-lo;rg=sub(cg,lg);}
            if(cumulative>hi){hi=cumulative;hg=cg;}
            if(cumulative<lo){lo=cumulative;lg=cg;}
        }
        double capacity_f=e-fall;Grad capacity_g={-fg[0],-fg[1],1-fg[2]};
        if(e+cumulative-rise<capacity_f){capacity_f=e+cumulative-rise;capacity_g=sub(cg,rg);capacity_g[2]+=1;}
        double* row=out+12*y;
        row[0]=cumulative;row[4]=capacity_f;row[8]=power_min;
        for(int j=0;j<3;j++){row[1+j]=cg[j];row[5+j]=capacity_g[j];row[9+j]=pg[j];}
    }
}

// The input uses one fixed MW basis across all weather years. Storage energy
// is also fixed, with AC charge/discharge power E/duration. Every year closes.
struct Scan { double required=0, balance=0, peak_deficit=0, peak_surplus=0; };
static Scan scan(const float* d,const float* g,const long long* ends,int ny,
                 double k,double power,double eta) {
    Scan result;result.balance=std::numeric_limits<double>::infinity();
    const double leg=std::sqrt(eta);
    for(int y=0;y<ny;y++) {
        double cumulative=0,hi=0,lo=0,fall=0,rise=0;
        for(long long t=ends[y];t<ends[y+1];t++) {
            const double net=k*g[t]-d[t];
            result.peak_deficit=std::max(result.peak_deficit,-net);
            result.peak_surplus=std::max(result.peak_surplus,net);
            cumulative+=net>=0?std::min(net,power)*leg:net/leg;
            fall=std::max(fall,hi-cumulative);rise=std::max(rise,cumulative-lo);
            hi=std::max(hi,cumulative);lo=std::min(lo,cumulative);
        }
        result.required=std::max({result.required,fall,rise-cumulative});
        result.balance=std::min(result.balance,cumulative);
    }
    return result;
}

extern "C" void gf_requirement(const float* d,const float* g,const long long* ends,
 int ny,double k,double duration,double eta,double* output) {
    Scan unlimited=scan(d,g,ends,ny,k,std::numeric_limits<double>::infinity(),eta);
    output[1]=unlimited.balance;
    if(unlimited.balance < -1e-7){output[0]=std::numeric_limits<double>::infinity();return;}
    double lo=std::max(unlimited.required,duration*unlimited.peak_deficit);
    if(duration==0){output[0]=lo;return;}
    auto feasible=[&](double e) {
        Scan s=scan(d,g,ends,ny,k,e/duration,eta);
        return s.balance>=-1e-7 && s.required<=e+1e-8 && s.peak_deficit<=e/duration+1e-8;
    };
    if(feasible(lo)){output[0]=lo;return;}
    double hi=std::max({lo*1.25,lo+0.1,0.1});
    const double absolute_hi=std::max(lo,duration*unlimited.peak_surplus)+1e-6;
    while(hi<absolute_hi && !feasible(hi))hi=std::min(absolute_hi,hi*1.5);
    if(!feasible(hi)){output[0]=std::numeric_limits<double>::infinity();return;}
    for(int step=0;step<30 && hi-lo>1e-6*std::max(1.,hi);step++) {
        double mid=(lo+hi)/2;
        if(feasible(mid))hi=mid;else lo=mid;
    }
    output[0]=hi;
}

extern "C" void gf_dispatch(const float* d,const float* g,const long long* ends,
 int ny,double k,double e,double duration,double eta,double* metrics,
 double* residual,double* stored) {
    const double leg=std::sqrt(eta),power=duration>0?e/duration:std::numeric_limits<double>::infinity();
    // Per-year columns: gap, max hourly gap, worst24, worst168, closure error,
    // initial SOC, final SOC, charge, discharge, curtailed generation.
    for(int y=0;y<ny;y++) {
        double soc=e,initial=0,gap=0,peak=0,charge=0,discharge=0,curtail=0;
        for(int repeat=0;repeat<2;repeat++) {
            if(repeat==1)initial=soc;
            for(long long t=ends[y];t<ends[y+1];t++) {
                double net=k*g[t]-d[t],unmet=0;
                if(net>=0) {
                    double c=std::max(0.,std::min({net,power,(e-soc)/leg}));soc+=c*leg;
                    if(repeat==1){charge+=c;curtail+=net-c;}
                } else {
                    double supply=std::max(0.,std::min({-net,power,soc*leg}));soc-=supply/leg;
                    unmet=std::max(0.,-net-supply);if(repeat==1)discharge+=supply;
                }
                if(repeat==1){gap+=unmet;peak=std::max(peak,unmet);residual[t]=unmet;stored[t]=soc;}
            }
        }
        double w24=0,w168=0,r24=0,r168=0;
        for(long long t=ends[y];t<ends[y+1];t++) {
            r24+=residual[t];r168+=residual[t];
            if(t-ends[y]>=24)r24-=residual[t-24];
            if(t-ends[y]>=168)r168-=residual[t-168];
            if(t-ends[y]>=23)w24=std::max(w24,r24);
            if(t-ends[y]>=167)w168=std::max(w168,r168);
        }
        double* out=metrics+10*y;
        out[0]=gap;out[1]=peak;out[2]=w24;out[3]=w168;out[4]=std::abs(soc-initial);
        out[5]=initial;out[6]=soc;out[7]=charge;out[8]=discharge;out[9]=curtail;
    }
}
