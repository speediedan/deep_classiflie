async function checkNew(latest_recs){
    let headers = {};
    const response = await fetch(latest_recs);
    for(const header of response.headers){
        console.log(`Name: ${header[0]}, Value:${header[1]}`);
        headers[header[0]] = header[1];
    }
    return headers;
    }

async function wait(ms) {
  return new Promise(resolve => {
    setTimeout(resolve, ms);
  });
}

function config_progbar(){
    let options = {
        "stroke": '#003566',
        "stroke-width": 20,
        "preset": "circle",
        "value": 0
        };
    let bar = new ldBar("#progress_bar", options);
    return bar;
}

async function readStreamResponse(response) {
    let pct_progress = 0;
    let bar = await config_progbar();
    const reader = await response.body.getReader();
    const contentLength = +response.headers.get('Content-Length');
    let receivedLength = 0;
    let chunks = [];
    while(true) {
        const {done, value} = await reader.read();
        if (done) {
            break;
        }
        chunks.push(value);
        receivedLength += value.length;
        pct_progress = Math.round((receivedLength/contentLength)*100);
        await bar.set(pct_progress);
        await wait(20);
    }
    let chunksAll = new Uint8Array(receivedLength);
    let position = 0;
    for(let chunk of chunks) {
        chunksAll.set(chunk, position);
        position += chunk.length;
    }
    return new TextDecoder("utf-8").decode(chunksAll);

}

function fetchTimeout(url, ms){
    const controller = new AbortController();
    var signal = controller.signal;
    const promise = fetch(url, { signal });
    const timeout = setTimeout(() => controller.abort(), ms);
    return promise.finally(() => clearTimeout(timeout));
}


async function loadJSON(latest_recs) {
    let response = {};
    try {
        response = await fetchTimeout(latest_recs[0], 10000);
    }
    catch(err) {
        console.log("Failed loading from primary gateway "+latest_recs[0]+" attempting to fetch from backup "+latest_recs[1]);
        await $("#progress_bar").toggle();
        //$( ".loading_msg" ).html( "<span class=\"footnotes\">Failed loading from primary gateway:<br/> "+latest_recs[0]+
        //    ".<br/>Attempting to fetch from backup <br/>"+
        //    latest_recs[1]+"...</span>" );
        try {
            response = await fetchTimeout(latest_recs[1], 10000);
        }
        catch(err) {
            await $("#progress_bar").toggle();
            console.log("Failed loading from backup gateway "+latest_recs[1]+". Falling back to local cache: "+latest_recs[2]);
            $( ".loading_msg" ).html( "<span class=\"footnotes\">Failed loading from backup gateway "+latest_recs[1]+
                ".<br/>Falling back to local cache <br/>"+
                latest_recs[2]+"...</span>" );
            response = await fetchTimeout(latest_recs[2], 10000);
        }
    }
    let result = await readStreamResponse(response);
    let curr_json = await JSON.parse(result);
    return curr_json
}

function dec_fmt( data, type, row ) {
    return data.toFixed(2);
}

async function add_table_func() {
    let curr_json = {};
    var num_fmt = ['', '.', 2, ''];
    const latest_recs = ["https://cloudflare-ipfs.com/ipns/predictions.deepclassiflie.org",
    "https://gateway.pinata.cloud/ipns/predictions.deepclassiflie.org", "/assets/dc_infsvc_pub_cache.json"];
    await $("#progress_bar").toggle(); // disable progress bar unless we fallback to pinata gateway
    curr_json = await loadJSON(latest_recs);
    var datatab = $('#curr_preds').DataTable( {
        data: curr_json,
        columns: [
        { data: 'claim_text',
          render: function ( data, type, row ) {
            return '<div class="claim"><a  href="'+row.transcript_url+'" target="_blank">'+data+'</a></div>';
          }
        },
        { data: 'prediction',
          render: function ( data, type, row ) {
          let claim_class = '',
          dtext = '';
          if (data == 0) {
              claim_class='nofalsehood';
              dtext = 'No Falsehood Label';
          }else{
              claim_class='falsehood';
              dtext = 'Falsehood Label';
          }
          return '<span class="'+claim_class+'">'+dtext+'</span>';
          }
        },
        { data: 'bucket_acc',
            render: $.fn.dataTable.render.number(...num_fmt),
            type: 'num'},
        { data: 'ppv', render: $.fn.dataTable.render.number(...num_fmt)},
        { data: 'npv', render: $.fn.dataTable.render.number(...num_fmt)},
        { data: 'ppr', render: $.fn.dataTable.render.number(...num_fmt)},
        { data: 'npr', render: $.fn.dataTable.render.number(...num_fmt)},
        { data: 'tp_ratio', render: $.fn.dataTable.render.number(...num_fmt)},
        { data: 'tn_ratio', render: $.fn.dataTable.render.number(...num_fmt)},
        { data: 'fp_ratio', render: $.fn.dataTable.render.number(...num_fmt)},
        { data: 'fn_ratio', render: $.fn.dataTable.render.number(...num_fmt)},
        { data: 't_date',
          render: function ( data, type, row ) {
            if ( type === 'display' || type === 'filter' ) {
                var d = new Date( data );
                return (d.getMonth()+1) + '/' + (d.getDate()) +'/'+ d.getFullYear();
            }
            return data;
          }
        }
    ],
        order: [[ 11, 'desc' ]],
        scrollY: 400,
        scrollX: 300
        } );
    return datatab;
}

async function load_latest(){
    await $(".dt_instructions").toggle();
    var datatab = await add_table_func();
    await $(".loading_msg").toggle();
    await $("#progress_bar").hide();
    await $(".dt_instructions").toggle();
    window.dispatchEvent(new Event('resize'));

}

$(document).ready( load_latest );