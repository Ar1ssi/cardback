// The editor's holofoil preview: TCG Live's own TPCi/Cards3D shaders (the extract's WebApp translation, served by
// gui.py under /api/foil/) drawing the render's front, white plate, etch and cold foil, the way the game and the
// ptcg-sim client (vibecodepokemontcg2 client/src/setup/foil/tcgl-foil-material.js, tcgl-foil-view.js,
// tcgl-foil-pose.mjs) draw them. Constant buffers are filled the way Unity fills them: $Globals from the
// MAT_Cards3D_* material, UnityPerCamera / UnityPerDraw / UnityPerFrame from the mesh and camera. The mesh turns
// with the pointer for shading but is rasterized face-on, so the foil lies exactly over the card image.
import * as THREE from "three";

const CARD_W = 6.3;
const CARD_H = 8.8;
const CARD_ASPECT = CARD_W / CARD_H;
const FOV_DEG = 30;
const CAMERA_DISTANCE = CARD_H / 2 / Math.tan(((FOV_DEG / 2) * Math.PI) / 180);
const FOIL_UV = { u0: 0.142107, u1: 0.857893, v0: 0, v1: 1 };   // the card's window of the square front
const TILT_DEG = 14;                                              // WebApp createFoilView, full pointer deflection
const DEG = Math.PI / 180;

let dataPromise = null;

/** shaders.json + materials.json, fetched once. */
function loadFoilData() {
  if (dataPromise) return dataPromise;
  const get = (file) => fetch(`/api/foil/${file}`).then((res) => {
    if (!res.ok) throw new Error(`${file}: HTTP ${res.status}`);
    return res.json();
  });
  dataPromise = Promise.all([get("shaders.json"), get("materials.json")]).then(([shaders, materials]) => {
    if (!shaders?.shaders || !materials?.materials) throw new Error("foil shaders/materials malformed");
    return { linear: !!shaders.linear, shaders: shaders.shaders, materials: materials.materials,
             textures: materials.textures ?? {} };
  });
  dataPromise.catch(() => { dataPromise = null; });
  return dataPromise;
}

// ---------------------------------------------------------------- textures
const WRAP = { repeat: THREE.RepeatWrapping, clamp: THREE.ClampToEdgeWrapping, mirror: THREE.MirroredRepeatWrapping,
               mirror_once: THREE.MirroredRepeatWrapping };
const loader = new THREE.TextureLoader();
const materialTextures = new Map();      // material textures stay for the session; card textures go with their view

function loadTexture(url, meta, linear) {
  let done;
  const ready = new Promise((resolve, reject) => { done = { resolve, reject }; });
  const texture = loader.load(url, () => done.resolve(), undefined, () => done.reject(new Error(`texture ${url} failed`)));
  texture.colorSpace = linear && meta.srgb !== false ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  texture.wrapS = WRAP[meta.wrapU] ?? THREE.RepeatWrapping;
  texture.wrapT = WRAP[meta.wrapV] ?? THREE.RepeatWrapping;
  if (meta.filter === "point") {
    texture.magFilter = THREE.NearestFilter;
    texture.minFilter = THREE.NearestFilter;
  } else {
    texture.anisotropy = 16;
  }
  ready.catch(() => {});
  return { texture, ready };
}

const plainTexture = (r, g, b) => {
  const texture = new THREE.DataTexture(new Uint8Array([r, g, b, 255]), 1, 1);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.needsUpdate = true;
  return texture;
};
const PLAIN = { white: plainTexture(255, 255, 255), gray: plainTexture(128, 128, 128), black: plainTexture(0, 0, 0),
                bump: plainTexture(128, 128, 255) };

const toLinear = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));

// ---------------------------------------------------------------- material
/** RawShaderMaterial for one printing: {front, shader, material, cc, textures: {wp, etch, cf}} (URLs or null). */
function createFoilMaterial(data, printing) {
  const sh = data.shaders[printing.shader];
  if (!sh) throw new Error(`foil shader ${printing.shader} not in the extract`);
  const mat = data.materials[printing.material]
    ?? Object.values(data.materials).find((m) => m.shader === printing.shader) ?? { tex: {}, floats: {}, colors: {} };

  // property values: shader defaults <- material <- the client's big-card view settings
  const vals = {};
  for (const [name, prop] of Object.entries(sh.props)) {
    vals[name] = prop.type === "vec" ? prop.default.slice() : prop.type === "float" ? prop.default : null;
  }
  Object.assign(vals, mat.floats ?? {}, mat.colors ?? {});
  if (data.linear) {            // Unity hands Color properties (not Vectors) to the shader converted from sRGB
    for (const [name, prop] of Object.entries(sh.props)) {
      const v = vals[name];
      if (prop.color && Array.isArray(v)) vals[name] = [toLinear(v[0]), toLinear(v[1]), toLinear(v[2]), v[3]];
    }
  }
  Object.assign(vals, {
    _OverallBrightness: 1, _FoilOnSmoothness: 0, _Smoothness: 0, _Glossiness: 0, _Metallic: 0.8, _NonFoilsheen: 0,
    _ShadowDarknessLimit: 0.7, _UseCCFoil: printing.cc ? 1 : 0, _LightDirection: [0, 0, -1, 0], _CardLighting_On: 0,
  });
  const light = vals._LightDirection;            // Unity is left-handed (cards face -Z): mirror the light's z
  vals._LightDirection = [light[0], light[1], -light[2], light[3] ?? 0];

  const cardTextures = [];
  const waits = [];
  const cardTexture = (url, fallback) => {
    if (!url) return fallback;
    const t = loadTexture(url, {}, data.linear);
    cardTextures.push(t.texture);
    waits.push(t.ready);
    return t.texture;
  };
  const materialTexture = (name) => {
    const meta = data.textures[name];
    if (!meta?.url) return null;
    const key = `${meta.url}|${data.linear}`;
    if (!materialTextures.has(key)) materialTextures.set(key, loadTexture(meta.url, meta, data.linear));
    const t = materialTextures.get(key);
    waits.push(t.ready);
    return t.texture;
  };
  // What Unity binds where a material sets no texture: the property's declared default. Etch means different
  // things per shader (SvHolo: white = not etched; FlatSilver: gray; SwHolo: black), so a card without one must
  // get its shader's default: plain white on a FlatSilver card darkens it to 73 % in linear light.
  const unbound = (prop) => {
    const def = sh.props[prop]?.default;
    return def === "black" ? PLAIN.black : def === "gray" ? PLAIN.gray : def === "bump" ? PLAIN.bump : PLAIN.white;
  };
  const textureFor = (prop) => {
    if (prop === "_CardColorDiffuse") return cardTexture(printing.front, PLAIN.gray);
    if (prop === "_CardEtch") return cardTexture(printing.textures.etch, unbound(prop));
    if (prop === "_CardWhitePlateMask") return cardTexture(printing.textures.wp, unbound(prop));
    if (prop === "_CardColdFoilMask") return cardTexture(printing.textures.cf, unbound(prop));
    if (prop === "_CCPatternTex" && printing.cc) {        // cast-and-cure pattern the card's manifest names
      const cc = materialTexture(printing.cc);
      if (cc) return cc;
    }
    const bound = mat.tex?.[prop]?.name;
    const fromMaterial = bound ? materialTexture(bound) : null;
    return fromMaterial ?? unbound(prop);
  };
  const scaleOffset = (prop) => {
    const t = mat.tex?.[prop];
    return t ? [t.scale[0], t.scale[1], t.offset[0], t.offset[1]] : [1, 1, 0, 0];
  };

  // constant buffers
  const uniforms = {};
  const buffers = [];
  for (const [stage, cbs] of [["v", sh.vsCB], ["f", sh.fsCB]]) {
    const src = stage === "v" ? sh.vs : sh.fs;
    for (const [slot, cb] of Object.entries(cbs)) {
      const declared = new RegExp(`uniform vec4 ${stage}cb${slot}\\[(\\d+)\\]`).exec(src);
      const length = declared ? Number(declared[1]) : Math.ceil(cb.size / 16);
      const arr = Array.from({ length }, () => new THREE.Vector4());
      uniforms[`${stage}cb${slot}`] = { value: arr };
      buffers.push({ arr, cb });
    }
    // builtin buffers a variant does not list: named from how the program reads them
    for (const match of src.matchAll(new RegExp(`uniform vec4 (${stage}cb\\d+)\\[(\\d+)\\]`, "g"))) {
      const [, name, size] = match;
      if (uniforms[name]) continue;
      const arr = Array.from({ length: Number(size) }, () => new THREE.Vector4());
      uniforms[name] = { value: arr };
      const buffer = new RegExp(`${name}\\[(17|18|19|20)\\]`).test(src) && Number(size) >= 21 ? "UnityPerFrame"
        : new RegExp(`${name}\\[0\\]\\.xyzw \\* v0\\.xxxx|${name}\\[3\\]\\.xyz \\* v0\\.www`).test(src) ? "UnityPerDraw"
          : "UnityPerCamera";
      buffers.push({ arr, cb: { buffer, params: [] } });
    }
  }
  const put = (arr, offset, values) => {
    for (let i = 0; i < values.length; i += 1) {
      const o = offset / 4 + i;
      const v = arr[o >> 2];
      if (v) v.setComponent(o & 3, values[i]);
    }
  };
  for (const { arr, cb } of buffers) {
    if (cb.buffer !== "$Globals") continue;
    for (const p of cb.params) {
      const v = p.name.endsWith("_ST") ? scaleOffset(p.name.slice(0, -3)) : vals[p.name];
      if (v == null) continue;
      put(arr, p.offset, Array.isArray(v) ? v.slice(0, p.dim) : [v]);
    }
  }
  for (const [slot, name] of Object.entries(sh.fsTex)) uniforms[`t${slot}`] = { value: textureFor(name) };

  const strip = (src) => src.replace(/^#version 300 es\s*/, "");
  const material = new THREE.RawShaderMaterial({
    vertexShader: strip(sh.vs), fragmentShader: strip(sh.fs), uniforms, glslVersion: THREE.GLSL3,
    side: THREE.FrontSide, depthWrite: true, transparent: false,
  });
  material.ready = Promise.all(waits);
  material.ready.catch(() => {});

  const rasterMatrix = new THREE.Matrix4();
  const inv = new THREE.Matrix4();
  const camPos = new THREE.Vector3();
  material.update = (mesh, camera, t, rasterVP) => {
    mesh.updateMatrixWorld();
    camera.updateMatrixWorld();
    rasterMatrix.fromArray(rasterVP);
    inv.copy(mesh.matrixWorld).invert();
    camera.getWorldPosition(camPos);
    const time = [t / 20, t, t * 2, t * 3];
    const sinTime = [Math.sin(t / 8), Math.sin(t / 4), Math.sin(t / 2), Math.sin(t)];
    const cosTime = [Math.cos(t / 8), Math.cos(t / 4), Math.cos(t / 2), Math.cos(t)];
    for (const { arr, cb } of buffers) {          // Unity's builtin buffers have a fixed layout
      if (cb.buffer === "UnityPerCamera") {
        put(arr, 0, time);
        put(arr, 16, sinTime);
        put(arr, 32, cosTime);
        put(arr, 48, [1 / 60, 60, 1 / 60, 60]);
        put(arr, 64, [camPos.x, camPos.y, camPos.z, 0]);
        put(arr, 80, [1, camera.near, camera.far, 1 / camera.far]);
        put(arr, 128, [0, 0, 0, 0]);
      } else if (cb.buffer === "UnityPerDraw") {
        put(arr, 0, mesh.matrixWorld.elements);
        put(arr, 64, inv.elements);
        put(arr, 128, [1, 1, 0, 0]);
        put(arr, 144, [0, 0, 0, 1]);
      } else if (cb.buffer === "UnityPerFrame") {
        put(arr, 80, camera.projectionMatrix.elements);
        put(arr, 144, camera.matrixWorldInverse.elements);
        put(arr, 208, camera.matrixWorld.elements);
        put(arr, 272, rasterMatrix.elements);
      }
    }
    const named = {
      _Time: time, _SinTime: sinTime, _CosTime: cosTime, _WorldSpaceCameraPos: [camPos.x, camPos.y, camPos.z],
      unity_ObjectToWorld: mesh.matrixWorld.elements, unity_WorldToObject: inv.elements,
      unity_MatrixVP: rasterMatrix.elements, unity_MatrixV: camera.matrixWorldInverse.elements,
      unity_MatrixInvV: camera.matrixWorld.elements, unity_MatrixP: camera.projectionMatrix.elements,
      unity_OrthoParams: [1, 1, 0, 0], unity_WorldTransformParams: [0, 0, 0, 1],
    };
    for (const { arr, cb } of buffers) {
      for (const p of cb.params) if (named[p.name]) put(arr, p.offset, named[p.name]);
    }
  };
  material.release = () => {
    material.dispose();
    for (const texture of cardTextures) texture.dispose();
  };
  return material;
}

/** Card face with rounded corners (63 x 88 mm, 3 mm radius) in the front's UV window, +x tangents. */
function foilGeometry(w, h, radius = w * 0.048, segments = 6) {
  const shape = new THREE.Shape();
  const x0 = -w / 2;
  const y0 = -h / 2;
  const r = radius;
  shape.moveTo(x0 + r, y0);
  shape.lineTo(x0 + w - r, y0);
  shape.quadraticCurveTo(x0 + w, y0, x0 + w, y0 + r);
  shape.lineTo(x0 + w, y0 + h - r);
  shape.quadraticCurveTo(x0 + w, y0 + h, x0 + w - r, y0 + h);
  shape.lineTo(x0 + r, y0 + h);
  shape.quadraticCurveTo(x0, y0 + h, x0, y0 + h - r);
  shape.lineTo(x0, y0 + r);
  shape.quadraticCurveTo(x0, y0, x0 + r, y0);
  const geometry = new THREE.ShapeGeometry(shape, segments);
  const pos = geometry.attributes.position;
  const { u0, u1, v0, v1 } = FOIL_UV;
  const uv = new Float32Array(pos.count * 2);
  const normal = new Float32Array(pos.count * 3);
  const tangent = new Float32Array(pos.count * 4);
  for (let i = 0; i < pos.count; i += 1) {
    uv[i * 2] = u0 + ((pos.getX(i) - x0) / w) * (u1 - u0);
    uv[i * 2 + 1] = v0 + ((pos.getY(i) - y0) / h) * (v1 - v0);
    normal.set([0, 0, 1], i * 3);
    tangent.set([1, 0, 0, 1], i * 4);
  }
  geometry.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  geometry.setAttribute("normal", new THREE.BufferAttribute(normal, 3));
  geometry.setAttribute("tangent", new THREE.BufferAttribute(tangent, 4));
  return geometry;
}

// ---------------------------------------------------------------- pose (column-major 4x4, three's layout)
function rotationMatrix(rx, ry) {             // R = Ry * Rx
  const cx = Math.cos(rx), sx = Math.sin(rx), cy = Math.cos(ry), sy = Math.sin(ry);
  return [cy, 0, -sy, 0, sy * sx, cx, cy * sx, 0, sy * cx, -sx, cy * cx, 0, 0, 0, 0, 1];
}

function multiply(a, b) {
  const out = new Array(16);
  for (let col = 0; col < 4; col += 1) {
    for (let row = 0; row < 4; row += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) sum += a[k * 4 + row] * b[col * 4 + k];
      out[col * 4 + row] = sum;
    }
  }
  return out;
}

const transposeRotation = (m) => [m[0], m[4], m[8], 0, m[1], m[5], m[9], 0, m[2], m[6], m[10], 0, 0, 0, 0, 1];
const clampUnit = (v) => (Number.isFinite(v) ? Math.min(Math.max(v, -1), 1) : 0);

// ---------------------------------------------------------------- the view
let renderer = null;
/**
 * Draws `printing` on `canvas` until stop(). tilt(): {x, y} in [-1, 1] (the pointer's deflection; x turns the
 * card about its vertical axis). onError(message): WebGL, shader or texture failure (the view stops itself).
 * -> Promise of {stop()}, resolved once every texture has loaded and the first frame is drawn.
 */
export async function showFoil(canvas, printing, tilt, onError) {
  const data = await loadFoilData();
  if (!renderer || renderer.domElement !== canvas) {
    renderer?.dispose();
    renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, premultipliedAlpha: false,
                                         preserveDrawingBuffer: true });
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    renderer.setClearColor(0x000000, 0);
  }
  let failed = null;
  renderer.debug.onShaderError = () => { failed = "the shader failed to compile"; };
  const material = createFoilMaterial(data, printing);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(FOV_DEG, CARD_ASPECT, 0.1, 100);
  camera.position.set(0, 0, CAMERA_DISTANCE);
  camera.lookAt(0, 0, 0);
  const mesh = new THREE.Mesh(foilGeometry(CARD_W, CARD_H), material);
  mesh.matrixAutoUpdate = false;
  mesh.frustumCulled = false;
  scene.add(mesh);
  await material.ready;

  let running = true;
  let raf = 0;
  const stop = () => {
    running = false;
    cancelAnimationFrame(raf);
    material.release();
    mesh.geometry.dispose();
  };
  const draw = (nowMs) => {
    if (!running) return;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
    const { x, y } = tilt();
    const rotation = rotationMatrix(-clampUnit(y) * TILT_DEG * DEG, clampUnit(x) * TILT_DEG * DEG);
    const stretchX = camera.aspect / CARD_ASPECT;     // our cards run 733:1024, the mesh 63:88
    mesh.matrix.fromArray(multiply(rotation, [stretchX, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]));
    mesh.matrixWorldNeedsUpdate = true;
    const rasterVP = multiply(multiply(camera.projectionMatrix.elements, camera.matrixWorldInverse.elements),
                              transposeRotation(rotation));
    material.update(mesh, camera, nowMs / 1000, rasterVP);
    renderer.render(scene, camera);
    if (failed) {
      stop();
      onError?.(failed);
      return;
    }
    raf = requestAnimationFrame(draw);
  };
  draw(performance.now());
  return { stop };
}
