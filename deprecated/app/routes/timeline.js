import Route from '@ember/routing/route';
import { csv } from 'd3-fetch';
// import d3 from 'd3';

export default Route.extend({
  model() {
    // return new Promise(function(resolve, reject) {
    //   $.ajax({
    //     dataType: "json",
    //     url: 'assets/timeline-dump.json',
    //     success: function (data){ 
    //       console.log(data)
    //       resolve(data)
    //     },
    //     error: function(err) {
    //       reject(err)
    //     }
    //   })
    // }).then(function (x) {
    //   return x //.map(function (y) { return y.toJSON(); })
    // });
    // console.log("---" + y)
    // console.log(Ember.Object.create({src: 'assets/timeline.csv'}))
    // console.log("asdasd")
    // return this.get('store').findAll('timeline').then(function (x) {
    //   return x.map(function (y) { return y.toJSON(); })
    // });

    const kfreq = (1.8 * 1024 * 1024);

    return csv("assets/timeline.csv", function(d) {
      return {
        label     : d.thread_id,// + "::" + d.operator,// + "::" + (+d.coreid % 2).toString(),
        timestamp : ((+d.timestamp) / kfreq),
        className : d.op,
        tid       : d.thread_id,
        content   : {"op": d.op, "core": d.coreid, "operator": d.operator}
      };
    })
    .then(function (data) {
      var out = []
      var d   = {}
      for (var i = 0 ; i < data.length ; ++i){
        if (+(data[i].className) / 2 >= 0){
          // if (+(data[i].className) / 2 == 6 / 2) continue;
          // if (+(data[i].className) / 2 == 7 / 2) continue;
          if ((+(data[i].className) % 2) == 1){
            var x = d[data[i].tid][data[i].className >> 1]
            data[i].end   = data[i].timestamp;
            data[i].start = x.timestamp;
            out.push(data[i])
          } else {
            if (!(data[i].tid in d)){
              d[data[i].tid] = {}
            }
            d[data[i].tid][data[i].className / 2] = data[i]
          }
        } else {
          data[i].start = data[i].timestamp;
          data[i].end   = data[i].timestamp + 0.0001;
          delete data[i].timestamp;

          // out.push(data[i])
        }
      }
      return out
    })
    .then(function (data) {
      let t = Math.min(...data.map(e => e.start));
      return data.map(e => {
        e.start = Math.round((e.start - t) * 100000)/100000;
        e.end   = Math.round((e.end   - t) * 100000)/100000;
        return e;
      });
    });
  //   .then(function(data) {
  //     var maxtimestamp = 0
  //     var mintimestamp = 1468963555040823 + 100000000

  //     var lnames = [];
  //     for (var i = 0 ; i < 65 ; ++i) lnames.push("color" + (i+1));
  //     var labels = {}
  //     var parent = {}
  //     var className = {}
  //     var ccount  = 0
  //     var count  = 0
  //     var tlines = {}
  //     data.forEach(function(x) {
  //       // if (x.className == 31) return;
  //       // if (x.className == 33) return;
  //       // if (x.className == 35) return;
  //     // for (var x in data){
  //       if (x.start < mintimestamp) mintimestamp = x.start;
  //       if (x.end   > maxtimestamp) maxtimestamp = x.end  ;

  //       if (tlines.hasOwnProperty(x.label)){
  //         tlines[x.label].push({
  //           "start" : x.start,
  //           "end"   : x.end,
  //           "className": x.className,
  //           "content": x.content
  //         })
  //       } else {
  //         tlines[x.label] = [{
  //           "start" : x.start,
  //           "end"   : x.end,
  //           "className": x.className,
  //           "content": x.content
  //         }]
  //         labels[x.label] = lnames[count++ % lnames.length];
  //       }

  //       if (!(x.className in className)){
  //         className[x.className] = lnames[ccount++ % lnames.length];
  //       }

  //       // var detailed_label = x.label + "::" + x.tid;
  //       // if (tlines.hasOwnProperty(detailed_label)){
  //       //   tlines[detailed_label].push({
  //       //     "start"  : x.start,
  //       //     "end"    : x.end,
  //       //     "className": x.className,
  //       //     "content": x.content
  //       //   })
  //       // } else {
  //       //   tlines[detailed_label] = [{
  //       //     "start"  : x.start,
  //       //     "end"    : x.end,
  //       //     "className": x.className,
  //       //     "content": x.content
  //       //   }]
  //       //   labels[detailed_label] = lnames[count++ % lnames.length];
  //       //   parent[detailed_label] = x.label
  //       // }
  //     })
  //     console.log(mintimestamp)
  //     console.log(maxtimestamp)
  //     var dlines = [];
  //     var ids    = {};
  //     var id_cnt = 0;
  //     Object.keys(tlines).forEach(function (label){
  //       // var s = []
  //       // for (var x in tlines[label]){
  //       //   if (x.start - mintimestamp > 15) continue;
  //       //   s.push({
  //       //     "start"  : x.start - mintimestamp,
  //       //     "end"    : x.end   - mintimestamp,
  //       //     "content": x.content,
  //       //     "className": className[x.className]
  //       //   })
  //       // }
  //       // if (s.length == 0) return
  //       dlines.push({
  //         "label": label,
  //         "start": 0,
  //         "end"  : maxtimestamp - mintimestamp,
  //         // "className": labels[label],
  //         // "sections": s
  //         "sections": tlines[label].map(function (x){
  //           return {
  //             "start"  : x.start - mintimestamp,
  //             "end"    : x.end   - mintimestamp,
  //             "content": x.dop,
  //             "className": "color" + (Math.floor((+x.className)/2) % lnames.length),// className[x.className]
  //           }
  //         })
  //       })
  //       if (label in parent){
  //         dlines[dlines.length - 1]["parent"] = parent[label];
  //       }
  //       ids[label] = id_cnt++;
  //     });
  //     dlines.forEach(function (x){
  //       if ("parent" in x){
  //         x.parent = ids[x.parent]
  //       }
  //     });
  //     console.log(dlines)
  //     return dlines; //.sort(function f(a, b){ return a.label < b.label; });
  //   });
  // },
  // modelr() {
  //   return [
  //     {
  //       "label"             : "Network Requests",
  //       "start"             : 1347800918,
  //       "end"               : 1347801818,
  //       "className"         : "network"
  //     },
  //     {
  //       "label"             : "Layout",
  //       "start"             : 1347801100,
  //       "end"               : 1347801918,
  //       "className"         : "layout"
  //     },
  //     {
  //       "label"             : "index.html",
  //       "start"             : 1347800918,
  //       "end"               : 1347801818,
  //       "className"         : "network",
  //       "parent"            : 0
  //     },
  //     {
  //       "label"             : "Paint",
  //       "start"             : 1347801118,
  //       "end"               : 1347801818,
  //       "className"         : "layout",
  //       "sections"          : [{
  //                               "start"     : 1347801118,
  //                               "end"       : 1347801218
  //                             }, {
  //                               "start"     : 1347801618,
  //                               "end"       : 1347801718
  //                             }],
  //       "parent"            : 1
  //     },
  //     {
  //       "label"             : "Reflow",
  //       "start"             : 1347801756,
  //       "end"               : 1347801907,
  //       "className"         : "layout",
  //       "parent"            : 1
  //     },
  //     {
  //       "label"             : "screen.css",
  //       "start"             : 1347801218,
  //       "end"               : 1347801618,
  //       "className"         : "network",
  //       "parent"            : 2
  //     },
  //     {
  //       "label"             : "app.js",
  //       "start"             : 1347801418,
  //       "end"               : 1347801818,
  //       "className"         : "network",
  //       "parent"            : 2
  //     },
  //     {
  //       "label"             : "JavaScript",
  //       "start"             : 1347801619,
  //       "end"               : 1347801920,
  //       "className"         : "javascript"
  //     },
  //     {
  //       "label"             : "domready",
  //       "start"             : 1347801664,
  //       "end"               : 1347801670,
  //       "className"         : "javascript",
  //       "parent"            : 7
  //     },
  //     {
  //       "label"             : "eval",
  //       "start"             : 1347801447,
  //       "end"               : 1347801920,
  //       "className"         : "javascript",
  //       "sections"          : [{
  //                               "start"     : 1347801447,
  //                               "end"       : 1347801497
  //                             }, {
  //                               "start"     : 1347801831,
  //                               "end"       : 1347801920
  //                             }],
  //       "parent"            : 7
  //     },
  //     {
  //       "label"             : "load",
  //       "start"             : 1347801820,
  //       "end"               : 1347801830,
  //       "className"         : "javascript",
  //       "parent"            : 7
  //     }
  //   ];
  }
});

// export default model3 = [
//       {
//         "label"             : "Network Requests",
//         "start"             : 1347800918,
//         "end"               : 1347801818,
//         "className"         : "network"
//       },
//       {
//         "label"             : "Layout",
//         "start"             : 1347801100,
//         "end"               : 1347801918,
//         "className"         : "layout"
//       },
//       {
//         "label"             : "index.html",
//         "start"             : 1347800918,
//         "end"               : 1347801818,
//         "className"         : "network",
//         "parent"            : 0
//       },
//       {
//         "label"             : "Paint",
//         "start"             : 1347801118,
//         "end"               : 1347801818,
//         "className"         : "layout",
//         "sections"          : [{
//                                 "start"     : 1347801118,
//                                 "end"       : 1347801218
//                               }, {
//                                 "start"     : 1347801618,
//                                 "end"       : 1347801718
//                               }],
//         "parent"            : 1
//       },
//       {
//         "label"             : "Reflow",
//         "start"             : 1347801756,
//         "end"               : 1347801907,
//         "className"         : "layout",
//         "parent"            : 1
//       },
//       {
//         "label"             : "screen.css",
//         "start"             : 1347801218,
//         "end"               : 1347801618,
//         "className"         : "network",
//         "parent"            : 2
//       },
//       {
//         "label"             : "app.js",
//         "start"             : 1347801418,
//         "end"               : 1347801818,
//         "className"         : "network",
//         "parent"            : 2
//       },
//       {
//         "label"             : "JavaScript",
//         "start"             : 1347801619,
//         "end"               : 1347801920,
//         "className"         : "javascript"
//       },
//       {
//         "label"             : "domready",
//         "start"             : 1347801664,
//         "end"               : 1347801670,
//         "className"         : "javascript",
//         "parent"            : 7
//       },
//       {
//         "label"             : "eval",
//         "start"             : 1347801447,
//         "end"               : 1347801920,
//         "className"         : "javascript",
//         "sections"          : [{
//                                 "start"     : 1347801447,
//                                 "end"       : 1347801497
//                               }, {
//                                 "start"     : 1347801831,
//                                 "end"       : 1347801920
//                               }],
//         "parent"            : 7
//       },
//       {
//         "label"             : "load",
//         "start"             : 1347801820,
//         "end"               : 1347801830,
//         "className"         : "javascript",
//         "parent"            : 7
//       }
//     ];

