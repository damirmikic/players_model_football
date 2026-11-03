const PE=require('./pricing.js'); const ref=JSON.parse(require('fs').readFileSync(process.argv[2]));
let worst=0;
for(const r of ref){
  const M=PE.scoreMatrix(r.lh,r.la,r.rho), x=PE.oneXTwo(M), T=PE.dist(M,(i,j)=>i+j), G=PE.dist(M,(i,j)=>i-j);
  const h=PE.htft(r.lh,r.la,r.rho);
  const mine=[...x, PE.fairFromDist(T,-2.5).W, PE.fairFromDist(G,-0.25).odds, PE.fairFromDist(G,0.75,1).odds,
    1-M[0].reduce((a,b)=>a+b)-M.reduce((a,row)=>a+row[0],0)+M[0][0], M[1][1], h.htft['1/1'], ...PE.applyMargin(x,0.06)];
  mine.forEach((v,i)=>worst=Math.max(worst,Math.abs(v-r.v[i])));
}
console.log('cases',ref.length,'max abs diff',worst.toExponential(2));
