import pandas as pd
import plotly.express as px
import streamlit as st
from SPARQLWrapper import JSON, SPARQLWrapper
from streamlit_agraph import Config, Edge, Node, agraph

# Page Configuration
st.set_page_config(
    page_title="NFL Knowledge Graph Dashboard", page_icon="🏈", layout="wide"
)

# Fuseki SPARQL Endpoint Configuration
FUSEKI_ENDPOINT = (
    "http://localhost:3030/NFL/sparql"  # Update dataset name if needed
)


@st.cache_data
def run_sparql_query(query_str):
  """Helper function to execute SPARQL queries against Fuseki and return a DataFrame"""
  sparql = SPARQLWrapper(FUSEKI_ENDPOINT)
  sparql.setQuery(query_str)
  sparql.setReturnFormat(JSON)

  try:
    results = sparql.query().convert()
    bindings = results["results"]["bindings"]

    # Parse JSON results into a Pandas DataFrame
    data = []
    for row in bindings:
      item = {}
      for key in row:
        item[key] = row[key]["value"]
      data.append(item)
    return pd.DataFrame(data)
  except Exception as e:
    st.error(f"Error executing SPARQL query: {e}")
    return pd.DataFrame()


# --- Sidebar Navigation ---
st.sidebar.title("🏈 NFL Graph Dashboard")
page = st.sidebar.selectbox(
    "Choose a View",
    ["Player Performance Analytics", "Player Subgraph Visualizer"],
)

# ==========================================
# PAGE 1: PLAYER PERFORMANCE ANALYTICS
# ==========================================
if page == "Player Performance Analytics":
  st.title("Player Performance & Route Analytics")
  st.markdown(
      "Querying the Apache Jena Fuseki Knowledge Graph via SPARQL to build"
      " interactive dashboards."
  )

  # SPARQL Query to get aggregated player stats (Direct SUM without inline type casting)
  query = """
    PREFIX ex: <http://example.org/nfl/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?playerName ?routeRan ?passResult (COUNT(?play) as ?playCount) (SUM(?yards) as ?totalYards)
    WHERE {
        ?play a ex:PlayerPlay ;
              ex:hasPlayer ?player ;
              ex:routeRan ?routeRan ;
              ex:passResult ?passResult ;
              ex:yardsGained ?yards .
        
        ?player foaf:name ?playerName .
    }
    GROUP BY ?playerName ?routeRan ?passResult
    LIMIT 500
    """

  with st.spinner("Fetching data from Jena Fuseki..."):
    df = run_sparql_query(query)

  if df.empty:
    st.warning(
        "No data returned. Make sure Apache Jena Fuseki is running and your"
        f" dataset endpoint matches: `{FUSEKI_ENDPOINT}`"
    )
  else:
    # Convert numerical strings to numeric types for Pandas/Plotly
    df["totalYards"] = pd.to_numeric(df["totalYards"], errors="coerce")
    df["playCount"] = pd.to_numeric(df["playCount"], errors="coerce")

    # Interactive Filters
    routes = ["All"] + sorted([r for r in df["routeRan"].unique() if r])
    selected_route = st.selectbox("Filter by Route Run", options=routes)
    if selected_route != "All":
      df = df[df["routeRan"] == selected_route]

    # Key Performance Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Plays Analyzed", int(df["playCount"].sum()))
    col2.metric("Total Yards Gained", int(df["totalYards"].sum()))
    col3.metric("Unique Players", df["playerName"].nunique())

    st.markdown("---")

    # Plotly Visual 1: Total Yards Gained by Player and Pass Result
    fig_yards = px.bar(
        df,
        x="playerName",
        y="totalYards",
        color="passResult",
        barmode="group",
        title="Total Yards Gained by Player and Pass Result",
        labels={"playerName": "Player Name", "totalYards": "Total Yards"},
    )
    st.plotly_chart(fig_yards, use_container_width=True)

    # Plotly Visual 2: Play Count Distribution by Route
    fig_routes = px.pie(
        df,
        names="routeRan",
        values="playCount",
        title="Play Distribution by Route Type",
    )
    st.plotly_chart(fig_routes, use_container_width=True)

# ==========================================
# PAGE 2: PLAYER SUBGRAPH VISUALIZER
# ==========================================
elif page == "Player Subgraph Visualizer":
  st.title("Player Relationship Subgraph")
  st.markdown(
      "Visualize targeted subgraphs (Player $\\rightarrow$ Plays $\\rightarrow$"
      " Routes) directly from Jena Fuseki."
  )

  # Fetch available players for dropdown
  player_list_query = """
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT DISTINCT ?name WHERE { ?p foaf:name ?name } ORDER BY ?name LIMIT 100
    """
  players_df = run_sparql_query(player_list_query)

  if players_df.empty:
    st.warning("Could not load player list. Please check your Fuseki connection.")
  else:
    selected_player = st.selectbox(
        "Select Player to Inspect", options=players_df["name"].tolist()
    )

    if st.button("Generate Subgraph"):
      # Query specific player relationships
      subgraph_query = f"""
            PREFIX ex: <http://example.org/nfl/>
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>

            SELECT ?playerName ?routeRan ?passResult ?yards WHERE {{
                ?play a ex:PlayerPlay ;
                      ex:hasPlayer ?player ;
                      ex:routeRan ?routeRan ;
                      ex:passResult ?passResult ;
                      ex:yardsGained ?yards .
                
                ?player foaf:name ?playerName .
                FILTER(?playerName = "{selected_player}")
            }}
            LIMIT 25
            """

      sub_df = run_sparql_query(subgraph_query)

      if not sub_df.empty:
        nodes = []
        edges = []

        # Central player node
        nodes.append(
            Node(
                id=selected_player,
                label=selected_player,
                size=25,
                color="#FF4B4B",
            )
        )

        for idx, row in sub_df.iterrows():
          play_node_id = f"Play_{idx}_{row['routeRan']}"
          play_label = (
              f"Route: {row['routeRan']}\nResult: {row['passResult']} ({row['yards']} yds)"
          )

          # Play nodes
          nodes.append(
              Node(id=play_node_id, label=play_label, size=15, color="#1C83E1")
          )

          # Relationship edges
          edges.append(
              Edge(
                  source=selected_player,
                  target=play_node_id,
                  label="RAN_ROUTE",
              )
          )

        config = Config(
            width=900,
            height=650,
            directed=True,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=False,
        )

        agraph(nodes=nodes, edges=edges, config=config)
      else:
        st.info(f"No play records found for {selected_player}.")