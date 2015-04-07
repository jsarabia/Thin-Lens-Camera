/*
	Joseph Sarabia
	CAP 6721
	Homework 1
	1/13/14
*/

//document.getElementById("button").onclick = function () { alert('hello!'); };


// var sceneLoc = "Cornell_box_model.json";
// var type = 2;

var sceneLoc = "sampleSphere.json";
var type = 1;

var radius = 0;
var far = 10;

function main(){
	var seed0 = 0;
	var seed1 = 0;
	var maxint = 2147483647;
	seed0 = Math.random() * maxint;		
	if(seed0 < 2) seed0 = 2;
	seed1 = Math.random() * maxint;		
	if(seed1 < 2) seed1 = 2;

	var cl = WebCL.createContext ();
	var device = cl.getInfo(WebCL.CL_CONTEXT_DEVICES)[0];
	var cmdQueue = cl.createCommandQueue (device, 0);
	var programSrc = loadKernel("raytrace");
	var program = cl.createProgram(programSrc);
	try {
		program.build ([device], "");
	} catch(e) {
		alert ("Failed to build WebCL program. Error "
		   + program.getBuildInfo (device, WebCL.CL_PROGRAM_BUILD_STATUS)
		   + ":  " + program.getBuildInfo (device, WebCL.CL_PROGRAM_BUILD_LOG));
		throw e;
	}
	var kernelName = "raytrace";
	try {
		kernel = program.createKernel (kernelName);
	} catch(e){
		alert("No kernel with name:"+ kernelName+" is found.");
		throw e;
	}
		var scene = new Scene(sceneLoc);
	var canvas = document.getElementById("canvas");
	var width=canvas.width, height=canvas.height;
	var canvasContext=canvas.getContext("2d");
	var canvasContent = canvasContext.createImageData(width,height);
	var nPixels = width*height;
	var nChannels = 4;
	var pixelBufferSize = nChannels*nPixels;
	var pixelBuffer = cl.createBuffer(WebCL.CL_MEM_WRITE_ONLY,pixelBufferSize);
	var cameraBufferSize = 40;
	var cameraBuffer = cl.createBuffer(WebCL.CL_MEM_WRITE_ONLY, cameraBufferSize);
	// [eye,at,up,fov]
	var cameraBufferData = new Float32Array([0,0,1,0,0,0,0,1,0,90]);
	var cameraObj = scene.getViewSpec(0);
	if (cameraObj)
	{
		cameraBufferData[0] = cameraObj.eye[0];
		cameraBufferData[1] = cameraObj.eye[1];
		cameraBufferData[2] = cameraObj.eye[2];
		cameraBufferData[3] = cameraObj.at[0];
		cameraBufferData[4] = cameraObj.at[1];
		cameraBufferData[5] = cameraObj.at[2];
		cameraBufferData[6] = cameraObj.up[0];
		cameraBufferData[7] = cameraObj.up[1];
		cameraBufferData[8] = cameraObj.up[2];
		cameraBufferData[9] = cameraObj.fov;
	}

	if(type == 1){
		var sphereBufferSize = scene.getSphereBufferSize();
		var sphereBuffer = cl.createBuffer(WebCL.CL_MEM_WRITE_ONLY,(sphereBufferSize)?sphereBufferSize:1);
		var nSpheres = scene.getNspheres();
		var triangleBufferSize = sphereBufferSize;
		var triangleBuffer = sphereBuffer;
		var nTriangles = nSpheres;
	}
	else {
		var triangleBufferSize = scene.getTriangleBufferSize();
		var triangleBuffer = cl.createBuffer(WebCL.CL_MEM_WRITE_ONLY,(triangleBufferSize)?triangleBufferSize:1);
		var nTriangles = scene.getNtriangles();
		var sphereBufferSize = triangleBufferSize;
		var sphereBuffer = triangleBuffer;
		var nSpheres = nTriangles;
	}

	var nMaterials = scene.getNmaterials();
	var materialBufferSize = scene.getMaterialBufferSize();
	var materialBuffer = cl.createBuffer(WebCL.CL_MEM_WRITE_ONLY, (materialBufferSize)?materialBufferSize:1);

	kernel.setKernelArg(0,pixelBuffer);
	kernel.setKernelArg(1,cameraBuffer);
	kernel.setKernelArg(2,sphereBuffer);
	kernel.setKernelArg(3,triangleBuffer);
	kernel.setKernelArg(4,nTriangles,WebCL.types.UINT);
	kernel.setKernelArg(5,nSpheres,WebCL.types.UINT);
	kernel.setKernelArg(6,width,WebCL.types.UINT);
	kernel.setKernelArg(7,height,WebCL.types.UINT);
	kernel.setKernelArg(8,type, WebCL.types.UINT);
	kernel.setKernelArg(9,nMaterials,WebCL.types.UINT);
	kernel.setKernelArg(10,materialBuffer);
	kernel.setKernelArg(11,radius,WebCL.types.UINT);
	kernel.setKernelArg(12,far,WebCL.types.UINT);
	kernel.setKernelArg(13,seed0,WebCL.types.UINT);
	kernel.setKernelArg(14,seed1,WebCL.types.UINT);



	var dim = 2;
	var maxWorkElements = kernel.getWorkGroupInfo(device,webCL.KERNEL_WORK_GROUP_SIZE);// WorkElements in ComputeUnit
	var xSize = Math.floor(Math.sqrt(maxWorkElements));
	var ySize = Math.floor(maxWorkElements/xSize);
	var localWS = [xSize, ySize];
	var globalWS = [Math.ceil(width/xSize)*xSize, Math.ceil(height/ySize)*ySize];

	if(type == 1){
		cmdQueue.enqueueWriteBuffer(sphereBuffer, false, 0, sphereBufferSize, scene.getSphereBufferData(), []);
		cmdQueue.enqueueWriteBuffer(triangleBuffer, false, 0, sphereBufferSize, scene.getSphereBufferData(), []);
	}
	else {
		cmdQueue.enqueueWriteBuffer(triangleBuffer, false, 0, triangleBufferSize, scene.getTriangleBufferData(), []);
		cmdQueue.enqueueWriteBuffer(sphereBuffer, false, 0, triangleBufferSize, scene.getTriangleBufferData(), []);
		cmdQueue.enqueueWriteBuffer(materialBuffer, false, 0, materialBufferSize, scene.getMaterialBufferData(), []);
	}
	

		
	
	console.log(materialBufferSize);
	cmdQueue.enqueueWriteBuffer(cameraBuffer, false, 0, cameraBufferSize, cameraBufferData, []);
	cmdQueue.enqueueNDRangeKernel(kernel,globalWS.length,[],globalWS,localWS,[]);
	// Must be done by pushing a Read request to the command queue
	cmdQueue.enqueueReadBuffer(pixelBuffer,false,0,pixelBufferSize,canvasContent.data,[]);
	cmdQueue.finish();
	canvasContext.putImageData(canvasContent,0,0);
	pixelBuffer.release();
	cameraBuffer.release();
	cmdQueue.release();
	kernel.release();
	program.release();
	cl.release();
}

function loadKernel(id){
  var kernelElement = document.getElementById(id);
  console.log(document.getElementById(id));
  var kernelSource = kernelElement.text;
  if (kernelElement.src != "") {
      var mHttpReq = new XMLHttpRequest();
      mHttpReq.open("GET", kernelElement.src, false);
      mHttpReq.send(null);
      kernelSource = mHttpReq.responseText;
  } 
  return kernelSource;
}

function switchstuff(){
	if(type == 1){
		type = 2;
		sceneLoc = "Cornell_box_model.json";
	}
	else {
		type = 1;
		sceneLoc = "sampleSphere.json";
	}
	main();
}
document.getElementById("clickMe").onclick = switchstuff;

function radiusHandler(){
	radius = document.getElementById("radius").value;
	main();
}

function farHandler(){
	far = document.getElementById("far").value;
	main();
}