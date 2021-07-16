//for npm only:
//import * as tf from '@tensorflow/tfjs';
//import {loadGraphModel} from '@tensorflow/tfjs-converter';

// Approx. noraml distribution
function normal_rand()
{
  let rand = 0;

  for (let i = 0; i < 6; i += 1) {
    rand += Math.random();
  }

  return rand / 6;
}

// Return a randomized vector of length size with values range (-1, 1).
function get_latents(size)
{
	let latents = new Array(size);
	
	for(let i=0; i<size; i++)
		latents[i] = normal_rand()*2-1;
	
	return latents;
}

// Return an array of vectors of length arraySize each the size of the src with 
// a randomized offset of max maxOffset between the last vector and the first.
// The intermediate vectors between first and last are linearly interpolated.
function generate_latents_range(src, arraySize, maxOffset)
{
	let latents_array = new Array(arraySize);
	latents_array[0] = src;
	
	latents_array[arraySize-1] = new Array(src.length);
	for(let i=0; i<src.length; i++)
		latents_array[arraySize-1][i] = src[i] + (normal_rand()*2-1) * maxOffset;
		
	for(let j=1; j<arraySize-1; j++)
		latents_array[j] = new Array(src.length);

	for(let i=0; i<src.length; i++)
	{
		let step = (latents_array[arraySize-1][i] - latents_array[0][i])/arraySize;
		for(let j=1; j<arraySize-1; j++)
			latents_array[j][i] = latents_array[j-1][i] + step;
	}
	
	return latents_array;
}

// Generate the image cache based on latents
var img_cache = {};
function getLatentCache(latent)
{
	let latent_hash = latent[0] * 10000.0 + latent[1];
	return latent_hash;
}

function addToImageCache(img, latent)
{
	if(!img)
		return false;
	
	let latent_hash = getLatentCache(latent);
	
//	if(!img_cache[latent_hash] || img_cache[latent_hash]==NO_IMAGE)
	img_cache[latent_hash] = img;
}

function getImageFromCache(latent)
{
	let latent_hash = getLatentCache(latent);
	
	if(img_cache[latent_hash])
		return img_cache[latent_hash];
	else
		return null;	
}

// Save a spot in the cache
var NO_IMAGE = 1001;
function registerLatentCache(latent)
{
	let latent_hash = getLatentCache(latent);

	if(!img_cache[latent_hash])
		img_cache[latent_hash] = NO_IMAGE;
}

function render_image(gan_img)
{
	let ctx = document.querySelector('.output_img').getContext('2d');
	ctx.putImageData(gan_img, 0, 0);
}

/*
// Run the neural network if it had finished processing the last latent, otherwise store it,
// and run it recursively. Do not use return value due to recursion.
var progan_promise_ready = true;
var latest_latent = null;
var progan_img, img;
var model = null;
var backend = null;

async function render_gan_image(msg)
{
	if(!progan_promise_ready)
	{
		latest_latent = latents_tensor;
		return false;
	}
	
	const MODEL_URL = '/progan/weights/model.json';
	if(backend != msg.data.backend)
	{
		backend = msg.data.backend;
		await tf.setBackend(backend);
		model = await tf.loadGraphModel(MODEL_URL);
	}	
	
	let latents = msg.data.latents;
	progan_promise_ready = false;
	latest_latent = null;
	document.querySelector(".progress-report").innerHTML = "Processing...";
	let t_start = performance.now();
//	progan_img = await model.executeAsync([tf.tensor([[]]), tf.tensor([latents])]);
	model.executeAsync([tf.tensor([[]]), tf.tensor([latents])]).then((progan_img) => {
		let t_end = performance.now();
		document.querySelector(".progress-report").innerHTML = "Ready.";
		
		// Update status
		document.getElementById("backend-name").innerHTML = tf.getBackend();
		document.getElementById("render-time").innerHTML = (t_end-t_start)/1000.0;
			
		img = tf.tidy(() => {
			let transpose_data = progan_img.transpose([0, 2, 3, 1]);
			
			let img_data = transpose_data.add(tf.scalar(1)).mul(tf.scalar(0.5)).clipByValue(0,1).squeeze();
		//	img = img_from_tensor[0].map(x => x.map(y => y.map(z => Math.min(Math.max(Math.round((z+1)/2*255), 0), 255))));

			return img_data; 
		});
		progan_promise_ready = true;
		
		tf.browser.toPixels(img, document.querySelector('.output_img')).then(() => {
			if(latest_latent)
				render_gan_image(latest_latent);
		});
	});

	return true;
}
//*/

async function gan_messenger(msg) 
{
	if(msg.data.gan_img)
	{
		if(msg.data.thread == 1)
		{
			render_image(msg.data.gan_img);
			gan_active = false;
			
			// Check if dialer has changed value and run the network again.
			let latents_idx = Math.floor(document.getElementById('dialer').value / nLatents);
			if(getLatentCache(dialerLatents[latents_idx]) != getLatentCache(msg.data.latents))
			{
				renderMain(dialerLatents[latents_idx]);
			}
		}

		addToImageCache(msg.data.gan_img, msg.data.latents);

		if(msg.data.thread != 1)
			renderNext(msg.data.latents, msg.data.thread);
//		await tf.browser.toPixels(tf.tensor(msg.data.gan_img), document.querySelector('.output_img')); // crashes firefox. Of course.
	}
	else if(msg.data.status)
	{
		document.getElementById("status-report-"+msg.data.thread).innerHTML = msg.data.status;
		
		if(msg.data.time)
		{
			avgTime.avg = (avgTime.avg * avgTime.items + msg.data.time) / (avgTime.items + 1);
			avgTime.items++;
			document.getElementById("avg-run").innerHTML = avgTime.avg.toFixed(2);
		}
	}
}

async function renderMain(currentLatent)
{
	// check cache:
	let imgBitmap = getImageFromCache(currentLatent);
	if(imgBitmap && imgBitmap != NO_IMAGE)
		render_image(imgBitmap);
	else if(!gan_active)	// have to redo even if it is being concurrently calculated.
	{
		let postData = {backend: document.getElementById("backend").value, 
						latents: currentLatent,
						thread: 1};
		
		gan_active = true;
		gan_renderer.postMessage(postData);
	}
}

async function renderNext(currentLatent, nThread)
{
	let postData = {backend: document.getElementById("backend").value, 
					thread: nThread};

	// Find the index of the current latent:
	let i;
	let current_hash = getLatentCache(currentLatent);
	for(i=0; i<nLatents; i++)
		if(getLatentCache(dialerLatents[i]) == current_hash)
			break;
			
	// check cache:
	for(let j=i+1; j<nLatents; j++)
	{
		postData.latents = dialerLatents[j];
		let imgBitmap = getImageFromCache(postData.latents);
		if(!imgBitmap)		
		{
			registerLatentCache(postData.latents);

			background_renderers[nThread-2].postMessage(postData);
			break;
		}
	}
}

var gan_active = false;
var avgTime = { avg: 0, items: 0};
var nLatents = 10;
var gan_renderer = new Worker("modules/render-gan.js");
gan_renderer.addEventListener("message", gan_messenger);

//var offscreen = new OffscreenCanvas(document.querySelector('.output_img').width, 
//								    document.querySelector('.output_img').height);	// firefox must enable gfx.offscreencanvas.enable in about:config
	// document.querySelector('.output_img').transferControlToOffscreen();  // Not supported by Firefox at all.
//gan_renderer.postMessage({ canvas: offscreen }, [offscreen]);
//gan_renderer.postMessage({ canvas: {width: document.querySelector('.output_img').width,
//									height: document.querySelector('.output_img').hegith}});


var background_renderers = [];

document.getElementById("threads").addEventListener("change", async (e) => {
	for(let i=0; i<e.currentTarget.value; i++)
		background_renderers[i] = new Worker("modules/render-gan.js");
	background_renderers.forEach((renderer)=> renderer.addEventListener("message", gan_messenger));

	var report_box = document.querySelector(".proto-tr");
	for(let i=0; i<background_renderers.length; i++)
	{
		let clone_report = report_box.cloneNode(true);
		clone_report.querySelector("span").innerHTML = "Thread-" + (i+2) +" Status:";
		clone_report.querySelector("div").id = "status-report-" + (i+2);
		
		report_box.parentElement.appendChild(clone_report);
	}
	
	e.currentTarget.disabled = true;
});
//*/

var first_latents = get_latents(512);
var	dialerLatents = generate_latents_range(first_latents, nLatents, 0.5);

document.getElementById("backend").addEventListener("change", async (e) => {
//	await tf.setBackend(e.currentTarget.value);
//	let img = tf.browser.fromPixels(document.querySelector('.progan-pic img'));
//	tf.browser.toPixels(img, document.querySelector('.output_img'));
	const dialerElement = document.getElementById('dialer');
	dialerElement.value = 0;
	renderMain(dialerLatents[0]);
	
	dialerElement.addEventListener('input', (e) => {
		let dialerSetting = Math.floor(e.currentTarget.value / nLatents);
		renderMain(dialerLatents[dialerSetting]);
	});
	
	// Generate the next image in a background thread.
	for(let i=0; i<background_renderers.length; i++)
	{
		renderNext(dialerLatents[i], 2+i);	
	}
});
