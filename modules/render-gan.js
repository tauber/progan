// below code is under worker environment
// to import tfjs into worker from a cdn

importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js");
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/tf-backend-wasm.js");

// Run the neural network if it had finished processing the last latent, otherwise store it,
// and run it recursively. Do not use return value due to recursion.
var progan_promise_ready = true;
var progan_img, img, imgBitmap;
var model = null;
var backend = null;
var offscreen_canvas = null;
var postMsg = { thread: 0,
				status: ""
};

// return a Uint8ClampedArray for a canvas ImageData representing the input image.
function generateImageData(img)
{
	let h = img.length;
	let w = img[0].length;
	clamped_img = new Uint8ClampedArray(w * h * 4);
	for(let i=0; i<h; i++)
		for(let j=0; j<w; j++)
		{
			let idx = i*w*4 + j*4;
			clamped_img[idx] = img[i][j][0];
			clamped_img[idx+1] = img[i][j][1];
			clamped_img[idx+2] = img[i][j][2];
			clamped_img[idx+3] = 255;
		}
		
	return clamped_img;
}

onmessage = async function render_gan_image(msg)
{
	if(msg.data.thread) 
	{
		postMsg.thread = msg.data.thread;
		postMsg.status = "Progan network starting...";
		postMessage(postMsg);
	}
		
	if(msg.data.canvas)
	{
		offscreen_canvas = new OffscreenCanvas(msg.data.canvas.width, msg.data.canvas.height);
		postMsg.status = "Canvas set to " + offscreen_canvas;
		postMessage(postMsg);

		return true;
	}
	
	if(!progan_promise_ready)
	{
		postMsg.status = "This latent will be executed when the network is ready";
		postMessage(postMsg);
		return false;
	}

	const MODEL_URL = '/progan/weights/model.json';
	if(backend != msg.data.backend)
	{
		backend = msg.data.backend;
		if(backend == "wasm")
		{
			tf.env().set('WASM_HAS_MULTITHREAD_SUPPORT', false);
//			tf.env().set('WASM_HAS_SIMD_SUPPORT', false);
		}
		postMsg.status = "Setting backend to " + backend;
		postMessage(postMsg);
		await tf.setBackend(backend);
		postMsg.status = "Downloading network model (refreshing the browser for new faces will use cache)...";
		postMessage(postMsg);
		model = await tf.loadGraphModel(MODEL_URL);
	}	
	
	let latents = msg.data.latents;
	progan_promise_ready = false;
	
	postMsg.status = "Starting inference now...";
	postMessage(postMsg);
	let t_start = performance.now();
	progan_img = await model.executeAsync([tf.tensor([[]]), tf.tensor([latents])]);
	let t_end = performance.now();

	postMsg.time = (t_end-t_start)/1000.0;
	postMsg.status = "The inference time with backend " + tf.getBackend() + " is: " + postMsg.time + " seconds.";
	postMessage(postMsg);
	// reset the time for the other messages.
	postMsg.time = 0;

	img = tf.tidy(() => {
		let transpose_data = progan_img.transpose([0, 2, 3, 1]);
		
		let img_data = transpose_data.add(tf.scalar(1)).mul(tf.scalar(127.5)).clipByValue(0,255).squeeze();
	//	img = img_from_tensor[0].map(x => x.map(y => y.map(z => Math.min(Math.max(Math.round((z+1)/2*255), 0), 255))));

		return img_data; 
	});
	
	img = img.arraySync();
	imgBitmap = new ImageData(generateImageData(img), img.length);
			
	progan_promise_ready = true;

	postMessage({gan_img: imgBitmap, latents: latents, thread: postMsg.thread});
	
/*	
	if(!offscreen_canvas)
	{
		postMessage({status: "Please set the offscreen canvas dimentions first..."});
	}

	tf.browser.toPixels(img, offscreen_canvas).then(() => {
		console.log(img);
		imgBitmap = offscreen_canvas.transferToImageBitmap();
		addToImageCache(imgBitmap, latents);
		postMessage({gan_bitmap: imgBitmap });
		if(latest_latent)
			render_gan_image(latest_latent);
	});

*/
	return true;
}
