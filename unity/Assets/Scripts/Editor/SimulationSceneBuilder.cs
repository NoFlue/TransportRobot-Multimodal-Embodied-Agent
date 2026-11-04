using System.Linq;
using MultimodalAgent;
using Unity.AI.Navigation;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.AI;

/// <summary>
/// Simulation Scene을 코드로 구성한다: 울퉁불퉁한 Terrain, 장애물, 석탄·철·화로, 에이전트, 행동 서버.
/// 메뉴 Tools/Carrier Agent 또는 -executeMethod로 실행한다.
/// </summary>
public static class SimulationSceneBuilder
{
    const string ScenePath = "Assets/Scenes/Simulation Scene.unity";
    const string GeneratedFolder = "Assets/Scenes/Simulation Scene Generated";
    const string ModelFolder = "Assets/Prefabs/low-poly ores/";
    const string PlayerPath = "Builds/CarrierAgent/CarrierAgent.exe";
    const float TerrainSize = 60f;
    const float TerrainHeight = 6f;
    const int CarriedLayer = 2; // Ignore Raycast: NavMesh 굽기와 바닥 raycast에서 빠진다.
    static readonly string[] GeneratedRoots = { "World", "Items", "CarrierAgent", "AgentRuntime" };

    [MenuItem("Tools/Carrier Agent/Build Simulation Scene")]
    public static void Build()
    {
        var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
        foreach (var root in scene.GetRootGameObjects())
        {
            if (GeneratedRoots.Contains(root.name)) Object.DestroyImmediate(root);
        }
        if (AssetDatabase.IsValidFolder(GeneratedFolder)) AssetDatabase.DeleteAsset(GeneratedFolder);
        AssetDatabase.CreateFolder("Assets/Scenes", "Simulation Scene Generated");

        var world = new GameObject("World");
        var terrain = CreateTerrain(world.transform);
        CreateObstacles(world.transform, terrain);
        CreateFurnace(world.transform, terrain, new Vector2(0f, 4f));
        var surface = world.AddComponent<NavMeshSurface>();
        surface.collectObjects = CollectObjects.Children;
        surface.useGeometry = NavMeshCollectGeometry.PhysicsColliders;

        var items = new GameObject("Items").transform;
        var coal = CreateMaterial("Coal", new Color(0.07f, 0.07f, 0.08f), 0.35f, 0f);
        var iron = CreateMaterial("Iron Ore", new Color(0.62f, 0.36f, 0.24f), 0.45f, 0.5f);
        CreateItem(items, terrain, "coal_1", "Coal Chunk", coal, new Vector2(-17f, 14f));
        CreateItem(items, terrain, "coal_2", "Coal Chunk", coal, new Vector2(16f, -11f));
        CreateItem(items, terrain, "iron_1", "Iron Ore Chunk", iron, new Vector2(17f, 15f));
        CreateItem(items, terrain, "iron_2", "Iron Ore Chunk", iron, new Vector2(-16f, -9f));

        var agent = CreateAgent(terrain, new Vector2(0f, -22f));
        var runtimeObject = new GameObject("AgentRuntime");
        var runtime = runtimeObject.AddComponent<CarrierAgentRuntime>();
        SetReference(runtime, "agent", agent);
        SetReference(runtime, "itemsRoot", items);
        var chat = runtimeObject.AddComponent<AgentChatPanel>();
        SetReference(chat, "runtime", runtime);

        SetupOverviewCamera(scene);
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
        AssetDatabase.SaveAssets();
        Debug.Log("Simulation Scene built.");
    }

    [MenuItem("Tools/Carrier Agent/Build Windows Player")]
    public static void BuildWindowsPlayer()
    {
        var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
        {
            scenes = new[] { ScenePath },
            locationPathName = PlayerPath,
            target = BuildTarget.StandaloneWindows64,
            options = BuildOptions.None,
        });
        Debug.Log($"Player build {report.summary.result}: {report.summary.totalErrors} errors, output {PlayerPath}");
        if (Application.isBatchMode && report.summary.result != BuildResult.Succeeded) EditorApplication.Exit(1);
    }

    // ---------------- Terrain ----------------

    static Terrain CreateTerrain(Transform parent)
    {
        var data = new TerrainData { heightmapResolution = 129 };
        data.size = new Vector3(TerrainSize, TerrainHeight, TerrainSize);
        var resolution = data.heightmapResolution;
        var heights = new float[resolution, resolution];
        for (var z = 0; z < resolution; z++)
        {
            for (var x = 0; x < resolution; x++)
            {
                var u = x / (float)(resolution - 1);
                var v = z / (float)(resolution - 1);
                heights[z, x] = Mathf.PerlinNoise(u * 6f + 3.1f, v * 6f + 7.7f) * 0.08f +
                                Mathf.PerlinNoise(u * 17f + 11f, v * 17f + 5f) * 0.03f;
            }
        }
        data.SetHeights(0, 0, heights);

        var groundTexture = CreateGroundTexture();
        var layer = new TerrainLayer
        {
            diffuseTexture = groundTexture,
            tileSize = new Vector2(6f, 6f),
            smoothness = 0f,
            metallic = 0f,
        };
        AssetDatabase.CreateAsset(layer, GeneratedFolder + "/Ground.terrainlayer");
        data.terrainLayers = new[] { layer };
        AssetDatabase.CreateAsset(data, GeneratedFolder + "/Terrain.asset");

        var terrainObject = Terrain.CreateTerrainGameObject(data);
        terrainObject.name = "Terrain";
        terrainObject.transform.SetParent(parent, false);
        terrainObject.transform.position = new Vector3(-TerrainSize / 2f, 0f, -TerrainSize / 2f);
        return terrainObject.GetComponent<Terrain>();
    }

    static Texture2D CreateGroundTexture()
    {
        const int size = 128;
        var texture = new Texture2D(size, size, TextureFormat.RGBA32, true) { wrapMode = TextureWrapMode.Repeat };
        for (var y = 0; y < size; y++)
        {
            for (var x = 0; x < size; x++)
            {
                var n = Mathf.PerlinNoise(x * 0.09f, y * 0.09f) * 0.6f + Mathf.PerlinNoise(x * 0.4f, y * 0.4f) * 0.4f;
                var color = Color.Lerp(new Color(0.36f, 0.30f, 0.22f), new Color(0.52f, 0.45f, 0.34f), n);
                color.a = 0f; // URP Terrain은 diffuse 알파를 smoothness로 쓴다. 0이면 반사 없는 흙바닥.
                texture.SetPixel(x, y, color);
            }
        }
        texture.Apply();
        AssetDatabase.CreateAsset(texture, GeneratedFolder + "/Ground.asset");
        return texture;
    }

    static Vector3 Ground(Terrain terrain, Vector2 xz)
    {
        var point = new Vector3(xz.x, 0f, xz.y);
        point.y = terrain.SampleHeight(point) + terrain.transform.position.y;
        return point;
    }

    // ---------------- 장애물과 객체 ----------------

    static void CreateObstacles(Transform parent, Terrain terrain)
    {
        var root = new GameObject("Obstacles").transform;
        root.SetParent(parent, false);
        (string model, float x, float z, float size, float yaw)[] obstacles =
        {
            ("Large Cave Boulder", -7f, -11f, 3.2f, 20f),
            ("Large Cave Boulder", 9f, -3f, 3.0f, 140f),
            ("Large Cave Boulder", -11f, 7f, 3.4f, 75f),
            ("Small Cave Boulder", 4f, -14f, 1.6f, 0f),
            ("Small Cave Boulder", -3f, -6f, 1.4f, 60f),
            ("Small Cave Boulder", 12f, 9f, 1.8f, 200f),
            ("Rockfall Debris Pile", -21f, -1f, 3.0f, 30f),
            ("Rockfall Debris Pile", 20f, 2f, 3.0f, 210f),
            ("Natural Rock Column", -6f, 19f, 3.5f, 0f),
            ("Natural Rock Column", 8f, 21f, 3.0f, 90f),
            ("Stalagmite Cluster", -18f, -18f, 2.5f, 0f),
            ("Wooden Mining Barrel", 6f, 10f, 1.1f, 0f),
            ("Supply Mining Crate", -6f, 11f, 1.2f, 30f),
            ("Timber Plank Stack", 13f, -19f, 2.0f, 45f),
            ("Minecart", 21f, -9f, 2.0f, 80f),
        };
        foreach (var (model, x, z, size, yaw) in obstacles)
        {
            var position = Ground(terrain, new Vector2(x, z)) + Vector3.down * 0.1f;
            PlaceModel(model, root, model, position, size, yaw, null);
        }
    }

    static void CreateItem(Transform parent, Terrain terrain, string label, string model, Material material, Vector2 xz)
    {
        var item = PlaceModel(model, parent, label, Ground(terrain, xz), 0.55f, Random.Range(0f, 360f), material);
        SetLayer(item, CarriedLayer);
        item.AddComponent<LabeledObject>().Configure(label, true);
    }

    static void CreateFurnace(Transform parent, Terrain terrain, Vector2 xz)
    {
        var stone = CreateMaterial("Furnace Stone", new Color(0.45f, 0.42f, 0.40f), 0.15f, 0f);
        var dark = CreateMaterial("Furnace Dark", new Color(0.16f, 0.15f, 0.15f), 0.2f, 0f);
        var fire = CreateMaterial("Furnace Fire", new Color(1f, 0.45f, 0.1f), 0.5f, 0f);
        fire.EnableKeyword("_EMISSION");
        fire.SetColor("_EmissionColor", new Color(1f, 0.4f, 0.05f) * 3f);
        fire.globalIlluminationFlags = MaterialGlobalIlluminationFlags.RealtimeEmissive;

        var furnace = new GameObject("furnace_1");
        furnace.transform.SetParent(parent, false);
        AddBlock(furnace.transform, PrimitiveType.Cube, stone, new Vector3(0f, 0.8f, 0f), new Vector3(2.0f, 1.6f, 2.0f));
        AddBlock(furnace.transform, PrimitiveType.Cube, dark, new Vector3(0f, 1.72f, 0f), new Vector3(2.2f, 0.25f, 2.2f));
        AddBlock(furnace.transform, PrimitiveType.Cube, fire, new Vector3(0f, 0.6f, -1.01f), new Vector3(0.9f, 0.6f, 0.05f));
        AddBlock(furnace.transform, PrimitiveType.Cylinder, dark, new Vector3(0.5f, 2.5f, 0.5f), new Vector3(0.5f, 0.75f, 0.5f));
        var glow = new GameObject("Glow").AddComponent<Light>();
        glow.transform.SetParent(furnace.transform, false);
        glow.transform.localPosition = new Vector3(0f, 0.7f, -1.6f);
        glow.type = LightType.Point;
        glow.color = new Color(1f, 0.55f, 0.2f);
        glow.intensity = 3f;
        glow.range = 5f;

        var box = furnace.AddComponent<BoxCollider>();
        box.center = new Vector3(0f, 0.9f, 0f);
        box.size = new Vector3(2.2f, 1.8f, 2.2f);
        furnace.transform.position = Ground(terrain, xz);
        furnace.AddComponent<LabeledObject>().Configure("furnace_1", false);
    }

    static void AddBlock(Transform parent, PrimitiveType type, Material material, Vector3 position, Vector3 scale)
    {
        var block = GameObject.CreatePrimitive(type);
        Object.DestroyImmediate(block.GetComponent<Collider>());
        block.transform.SetParent(parent, false);
        block.transform.localPosition = position;
        block.transform.localScale = scale;
        block.GetComponent<Renderer>().sharedMaterial = material;
    }

    /// <summary>FBX를 지정한 크기로 맞추고 바닥 중심에 원점을 둔 뒤 BoxCollider를 붙인다.</summary>
    static GameObject PlaceModel(string model, Transform parent, string name, Vector3 position, float size, float yaw, Material material)
    {
        var root = new GameObject(name);
        root.transform.SetParent(parent, false);
        var asset = AssetDatabase.LoadAssetAtPath<GameObject>(ModelFolder + model + ".fbx");
        GameObject visual;
        if (asset != null)
        {
            visual = (GameObject)PrefabUtility.InstantiatePrefab(asset);
        }
        else
        {
            Debug.LogWarning($"Model {model} not found; using a cube.");
            visual = GameObject.CreatePrimitive(PrimitiveType.Cube);
        }
        foreach (var collider in visual.GetComponentsInChildren<Collider>()) Object.DestroyImmediate(collider);
        visual.transform.SetParent(root.transform, false);

        var bounds = RendererBounds(visual);
        var largest = Mathf.Max(bounds.size.x, bounds.size.y, bounds.size.z);
        if (largest > 1e-4f) visual.transform.localScale *= size / largest;
        bounds = RendererBounds(visual);
        visual.transform.position += new Vector3(-bounds.center.x, -bounds.min.y, -bounds.center.z);

        if (material != null)
        {
            foreach (var renderer in visual.GetComponentsInChildren<Renderer>())
            {
                renderer.sharedMaterials = Enumerable.Repeat(material, renderer.sharedMaterials.Length).ToArray();
            }
        }

        bounds = RendererBounds(visual);
        var box = root.AddComponent<BoxCollider>();
        box.center = bounds.center;
        box.size = bounds.size;
        root.transform.SetPositionAndRotation(position, Quaternion.Euler(0f, yaw, 0f));
        return root;
    }

    static Bounds RendererBounds(GameObject target)
    {
        var renderers = target.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0) return new Bounds(target.transform.position, Vector3.one);
        var bounds = renderers[0].bounds;
        foreach (var renderer in renderers) bounds.Encapsulate(renderer.bounds);
        return bounds;
    }

    // ---------------- 에이전트 ----------------

    static CarrierAgent CreateAgent(Terrain terrain, Vector2 xz)
    {
        var root = new GameObject("CarrierAgent");
        root.transform.SetPositionAndRotation(Ground(terrain, xz), Quaternion.identity);

        var body = CreateMaterial("Agent Body", new Color(0.95f, 0.75f, 0.15f), 0.4f, 0f);
        var visor = CreateMaterial("Agent Visor", new Color(0.1f, 0.12f, 0.15f), 0.8f, 0f);
        AddBlock(root.transform, PrimitiveType.Capsule, body, new Vector3(0f, 0.9f, 0f), new Vector3(0.8f, 0.9f, 0.8f));
        AddBlock(root.transform, PrimitiveType.Cube, visor, new Vector3(0f, 1.5f, 0.36f), new Vector3(0.5f, 0.15f, 0.12f));

        var eye = new GameObject("EyeCamera").AddComponent<Camera>();
        eye.transform.SetParent(root.transform, false);
        eye.transform.localPosition = new Vector3(0f, 1.6f, 0.5f);
        eye.transform.localRotation = Quaternion.Euler(20f, 0f, 0f);
        eye.fieldOfView = 70f;
        eye.nearClipPlane = 0.1f;
        eye.farClipPlane = 120f;
        eye.enabled = false;

        var hand = new GameObject("Hand").transform;
        hand.SetParent(root.transform, false);
        hand.localPosition = new Vector3(0.35f, 1.0f, 0.55f);
        var backpack = new GameObject("Backpack").transform;
        backpack.SetParent(root.transform, false);
        backpack.localPosition = new Vector3(0f, 1.2f, -0.45f);

        var nav = root.AddComponent<NavMeshAgent>();
        nav.radius = 0.4f;
        nav.height = 1.8f;
        nav.speed = 3.5f;
        nav.angularSpeed = 360f;
        nav.acceleration = 10f;
        nav.stoppingDistance = 0.1f;
        nav.enabled = false; // NavMesh를 런타임에 구운 뒤 켠다.

        var agent = root.AddComponent<CarrierAgent>();
        SetReference(agent, "eyeCamera", eye);
        SetReference(agent, "hand", hand);
        SetReference(agent, "backpack", backpack);
        SetLayer(root, CarriedLayer);
        return agent;
    }

    static void SetupOverviewCamera(UnityEngine.SceneManagement.Scene scene)
    {
        var main = scene.GetRootGameObjects().Select(o => o.GetComponent<Camera>()).FirstOrDefault(c => c != null && c.CompareTag("MainCamera"));
        if (main == null) return;
        main.transform.SetPositionAndRotation(new Vector3(0f, 38f, -40f), Quaternion.Euler(42f, 0f, 0f));
        main.farClipPlane = 200f;
    }

    // ---------------- 공통 ----------------

    static Material CreateMaterial(string name, Color color, float smoothness, float metallic)
    {
        var material = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        material.SetColor("_BaseColor", color);
        material.SetFloat("_Smoothness", smoothness);
        material.SetFloat("_Metallic", metallic);
        AssetDatabase.CreateAsset(material, $"{GeneratedFolder}/{name}.mat");
        return material;
    }

    static void SetReference(Object target, string field, Object value)
    {
        var serialized = new SerializedObject(target);
        serialized.FindProperty(field).objectReferenceValue = value;
        serialized.ApplyModifiedPropertiesWithoutUndo();
    }

    static void SetLayer(GameObject target, int layer)
    {
        foreach (var t in target.GetComponentsInChildren<Transform>(true)) t.gameObject.layer = layer;
    }
}
