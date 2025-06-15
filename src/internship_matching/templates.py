PALETTE = (
    "#1f77b4",  # BLUE
    "#ff7f0e",
    "#2ca02c",
    "#d62728",  # RED
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

THREEJS_TEMPLATE = (
    """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\"/>
  <title>3‑D t‑SNE – {title}</title>
  <style>
    body,html{{margin:0;height:100%;overflow:hidden;background:#111;color:#eee;font-family:sans-serif}}
    #container{{width:100%;height:100%}}
    #info{{position:fixed;top:0;left:0;padding:6px 8px;font-size:13px;background:rgba(0,0,0,0.6)}}
  </style>
  <script type=\"importmap\">
    {{
      \"imports\": {{
        \"three\": \"https://cdn.jsdelivr.net/npm/three@0.160/build/three.module.js\"
      }}
    }}
  </script>
</head>
<body>
<div id=\"container\"></div>
<div id=\"info\">Drag to rotate, scroll to zoom. Grey points = noise.</div>
<script type=\"module\">
import * as THREE from 'three';
import {{OrbitControls}} from 'https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/controls/OrbitControls.js';
const resp = await fetch('{json_path}');
const data = await resp.json();
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
camera.position.set(0,0,180);
const renderer = new THREE.WebGLRenderer({{ antialias:true }});
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('container').appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
// Axes
const axes = new THREE.AxesHelper(50);/*scene.add(axes);*/
// Points
const positions = new Float32Array(data.points.length*3);
const colors = new Float32Array(data.points.length*3);
function hexToRGB(hex){{const c=parseInt(hex.slice(1),16);return[(c>>16&255)/255,(c>>8&255)/255,(c&255)/255];}}
const labColorMap = new Map();
Object.entries(data.clusters).forEach(([cid,info],i)=>labColorMap.set(parseInt(cid),hexToRGB(info.color)));
for(let i=0;i<data.points.length;i++){{
  const p=data.points[i];
  positions[i*3]=p.x;positions[i*3+1]=p.y;positions[i*3+2]=p.z;
  const col=p.label==-1?[0.6,0.6,0.6]:labColorMap.get(p.label%{pal_len})||[1,1,1];
  colors[i*3]=col[0];colors[i*3+1]=col[1];colors[i*3+2]=col[2];
}}
const geo=new THREE.BufferGeometry();
geo.setAttribute('position',new THREE.BufferAttribute(positions,3));
geo.setAttribute('color',new THREE.BufferAttribute(colors,3));
const mat=new THREE.PointsMaterial({{size:0.1,vertexColors:true}});
scene.add(new THREE.Points(geo,mat));
// Hulls
for(const [cid,info] of Object.entries(data.clusters)){{
  if(cid=='-1') continue;
  const faceIdx=new Uint16Array(info.faces.flat());
  const verts=new Float32Array(info.vertices.flat());
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(verts,3));
  g.setIndex(new THREE.BufferAttribute(faceIdx,1));
  g.computeVertexNormals();
  const m=new THREE.MeshBasicMaterial({{color:info.color,transparent:true,opacity:0.25,side:THREE.DoubleSide}});
  scene.add(new THREE.Mesh(g,m));
}}
window.addEventListener('resize',()=>{{
  camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth,window.innerHeight);
}});
(function animate(){{requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);}})();
</script>
</body>
</html>"""
).replace("{pal_len}", str(len(PALETTE)))

THREEJS_PLOT_SIM_TEMPLATE = (
    """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>3-D Simulation – {title}</title>
  <style>
    body, html {{ margin:0; height:100%; overflow:hidden; background:#111; color:#eee; font-family:sans-serif }}
    #container {{ width:100%; height:100% }}
    #info {{ position:fixed; top:0; left:0; padding:6px 8px; font-size:13px; background:rgba(0,0,0,0.6) }}
  </style>
  <script type="importmap">
  {{
    "imports": {{
      "three": "https://cdn.jsdelivr.net/npm/three@0.160/build/three.module.js"
    }}
  }}
  </script>
</head>
<body>
  <div id="container"></div>
  <div id="info">Drag to rotate, scroll to zoom. Query = blue, matches = red, background = gray.</div>
  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/controls/OrbitControls.js';

    // load our JSON payload
    const resp = await fetch('{json_path}');
    const data = await resp.json();

    // scene & camera
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 1);  // start close to the query

    // renderer with antialias on
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.getElementById('container').appendChild(renderer.domElement);

    // controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // build points geometry
    const positions = new Float32Array(data.points.length * 3);
    const colors    = new Float32Array(data.points.length * 3);
    function hexToRGB(hex) {{
      const c = parseInt(hex.slice(1), 16);
      return [(c >> 16 & 255) / 255, (c >> 8 & 255) / 255, (c & 255) / 255];
    }}

    for (let i = 0; i < data.points.length; i++) {{
      const p = data.points[i];
      positions[i*3]   = p.x;
      positions[i*3+1] = p.y;
      positions[i*3+2] = p.z;
      const col = p.color.startsWith('#') 
        ? hexToRGB(p.color) 
        : [0.6, 0.6, 0.6];
      colors[i*3]   = col[0];
      colors[i*3+1] = col[1];
      colors[i*3+2] = col[2];
    }}

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
    const mat = new THREE.PointsMaterial({{ size: 0.1, vertexColors: true }});
    scene.add(new THREE.Points(geo, mat));

    // draw sphere around query point
    {{
      const query = data.points.find(p => p.size === 6); // query has size=6
      const matches = data.points.filter(p => p.size === 5);
      // compute max distance
      const dmax = Math.max(...matches.map(m => 
        Math.hypot(m.x - query.x, m.y - query.y, m.z - query.z)
      ));
      const sphereGeom = new THREE.SphereGeometry(dmax, 32, 32);
      const sphereMat  = new THREE.MeshBasicMaterial({{ 
        color: 0x00ff00, 
        transparent: true, 
        opacity: 0.2, 
        side: THREE.BackSide 
      }});
      const sphere = new THREE.Mesh(sphereGeom, sphereMat);
      sphere.position.set(query.x, query.y, query.z);
      scene.add(sphere);
    }}

    // lines from query to matches
    {{
      const query = data.points.find(p => p.size === 6);
      const matLines = new THREE.BufferGeometry();
      const verts = [];
      for (const p of data.points.filter(p => p.size === 5)) {{
        verts.push(query.x, query.y, query.z, p.x, p.y, p.z);
      }}
      matLines.setAttribute('position', new THREE.BufferAttribute(new Float32Array(verts), 3));
      const lineMat = new THREE.LineBasicMaterial({{ color: 0xff0000, linewidth: 1 }});
      scene.add(new THREE.LineSegments(matLines, lineMat));
    }}

    // resize handling
    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    // animate
    (function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }})();
  </script>
</body>
</html>"""
).replace("{pal_len}", str(len(PALETTE)))