set(LIGHTNING_APP_TARGETS)

function(add_lightning_app target)
    add_executable(${target}
            ${ARGN}
    )
    target_link_libraries(${target}
            ${PROJECT_NAME}.libs
            ${third_party_libs}
    )
    list(APPEND LIGHTNING_APP_TARGETS ${target})
    set(LIGHTNING_APP_TARGETS ${LIGHTNING_APP_TARGETS} PARENT_SCOPE)
endfunction()

if (ROBOCUP_LIGHTNING_BUILD_OFFLINE_TOOLS)
    add_lightning_app(run_frontend_offline
            run_frontend_offline.cc
    )

    add_lightning_app(run_loop_offline
            run_loop_offline.cc
    )

    add_lightning_app(run_slam_offline
            run_slam_offline.cc
    )
endif ()

add_lightning_app(run_slam_online
        run_slam_online.cc
)

add_lightning_app(run_loc_online
        run_loc_online.cc
)

if (ROBOCUP_LIGHTNING_BUILD_OFFLINE_TOOLS)
    add_lightning_app(run_loc_offline
            run_loc_offline.cc
    )

    add_lightning_app(convert_pcd_to_map
            convert_pcd_to_map.cc
    )

    if (ROBOCUP_LIGHTNING_WITH_PANGOLIN)
        add_lightning_app(test_ui
                test_ui.cc
        )
    endif ()
endif ()

install(TARGETS ${LIGHTNING_APP_TARGETS}
        DESTINATION lib/${PROJECT_NAME})
